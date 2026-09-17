"""Lease acquisition domain logic.

Concurrency design (all inside one READ COMMITTED transaction):

1. ``pg_advisory_xact_lock(hashtext(:key))``
   Transactions that carry the same idempotency key are serialised, so a lost
   response followed by a retry can never create a second lease.
2. Look up the stored idempotency record. Same parameters -> replay the
   original token/expiry; different parameters -> stable ``IDEMPOTENCY_CONFLICT``.
3. ``SELECT ... FROM antennas WHERE id = :antenna_id FOR UPDATE``
   Serialises every contender for the same antenna. Unknown antenna raises
   ``ANTENNA_NOT_FOUND`` before any row is written.
4. Look up an active lease with
   ``released_at IS NULL AND expires_at > clock_timestamp()``. A lease
   whose ``expires_at`` has been reached (``expires_at <= clock_timestamp()``)
   is gone: the boundary belongs to the new request. A lease released early
   is also immediately available for handover.
5. Insert the new lease (``expires_at = clock_timestamp() + make_interval``)
   and its idempotency record, then commit atomically.

Progress reporting and early release take the same antenna row lock as lease
acquisition. Renewal does too, and additionally serialises same-key retries on
an advisory lock before recording the extension in ``lease_renewals``. Every
timestamp originates from PostgreSQL; the host clock is never read.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.config import (
    MAX_LEASE_SECONDS,
    MAX_RENEW_SECONDS,
    MIN_LEASE_SECONDS,
    MIN_RENEW_SECONDS,
)
from app.errors import APIError


def _canonical_params(antenna_id: str, controller: str, duration_seconds: int) -> str:
    # Stable textual fingerprint; parameter names are part of it.
    return (
        f"antenna_id={antenna_id}\n"
        f"controller={controller}\n"
        f"duration_seconds={duration_seconds}"
    )


def _canonical_renew_params(lease_token: str, additional_seconds: int) -> str:
    # Same fingerprint discipline as acquisition, scoped to one lease token.
    return (
        f"lease_token={lease_token}\n"
        f"additional_seconds={additional_seconds}"
    )


def acquire_lease(
    conn: Connection,
    *,
    antenna_id: str,
    controller: str,
    duration_seconds: int,
    idempotency_key: str,
) -> dict[str, Any]:
    # Defence in depth: Pydantic validates the HTTP boundary, the service
    # validates any internal caller as well. Rejections happen before any
    # write statement is issued.
    if not (
        isinstance(duration_seconds, int)
        and MIN_LEASE_SECONDS <= duration_seconds <= MAX_LEASE_SECONDS
    ):
        raise APIError(
            422,
            "LEASE_DURATION_OUT_OF_RANGE",
            f"租期必须为 {MIN_LEASE_SECONDS} 至 {MAX_LEASE_SECONDS} 秒之间的整数。",
            {
                "duration_seconds": duration_seconds,
                "min": MIN_LEASE_SECONDS,
                "max": MAX_LEASE_SECONDS,
            },
        )

    fingerprint = _canonical_params(antenna_id, controller, duration_seconds)

    # 1. Serialise transactions sharing one idempotency key.
    conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": idempotency_key},
    )

    # 2. Replay or stable conflict.
    existing = conn.execute(
        text(
            """
            SELECT lease_id, request_params
            FROM idempotency_keys
            WHERE idempotency_key = :key
            """
        ),
        {"key": idempotency_key},
    ).mappings().first()

    if existing is not None:
        if existing.request_params != fingerprint:
            raise APIError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "同一幂等键曾用于不同的请求参数，拒绝执行。",
                {
                    "idempotency_key": idempotency_key,
                    "original_params": existing.request_params,
                    "request_params": fingerprint,
                },
            )
        replay = conn.execute(
            text(
                """
                SELECT id AS lease_id, antenna_id, controller,
                       token AS lease_token, acquired_at, expires_at
                FROM leases
                WHERE id = :lease_id
                """
            ),
            {"lease_id": existing.lease_id},
        ).mappings().first()
        # lease_id is NOT NULL with an FK; the row always exists.
        return {**dict(replay), "replay": True}

    # 3. Lock the antenna row (also proves the antenna is provisioned).
    antenna = conn.execute(
        text("SELECT id FROM antennas WHERE id = :antenna_id FOR UPDATE"),
        {"antenna_id": antenna_id},
    ).first()
    if antenna is None:
        raise APIError(
            404,
            "ANTENNA_NOT_FOUND",
            f"未知天线：{antenna_id}",
            {"antenna_id": antenna_id},
        )

    # 4. An unexpired lease wins; expiry boundary (expires_at == now) goes
    #    to the new request because the predicate is strictly greater-than.
    active = conn.execute(
        text(
            """
            SELECT token, expires_at
            FROM leases
            WHERE antenna_id = :antenna_id
              AND released_at IS NULL
              AND expires_at > clock_timestamp()
            ORDER BY acquired_at DESC, id DESC
            LIMIT 1
            """
        ),
        {"antenna_id": antenna_id},
    ).mappings().first()
    if active is not None:
        raise APIError(
            409,
            "ANTENNA_BUSY",
            f"天线 {antenna_id} 已被未到期租约占用。",
            {
                "antenna_id": antenna_id,
                "held_by_lease": active.token,
                "expires_at": active.expires_at.isoformat(),
            },
        )

    # 5. Create lease + idempotency record atomically. Token comes from
    #    PostgreSQL's CSPRNG so it is unpredictable on the wire. Standard
    #    base64 contains '/', '+' and '=' which are unsafe in a single URL
    #    path segment, so emit the base64url alphabet with padding stripped
    #    (43 chars for 32 random bytes).
    row = conn.execute(
        text(
            """
            WITH new_lease AS (
                INSERT INTO leases (antenna_id, controller, token, acquired_at, expires_at)
                VALUES (
                    :antenna_id,
                    :controller,
                    rtrim(
                        replace(
                            replace(encode(gen_random_bytes(32), 'base64'), '+', '-'),
                            '/', '_'
                        ),
                        '='
                    ),
                    clock_timestamp(),
                    clock_timestamp() + make_interval(secs => :duration)
                )
                RETURNING id AS lease_id, antenna_id, controller,
                          token AS lease_token, acquired_at, expires_at
            ), recorded AS (
                INSERT INTO idempotency_keys (idempotency_key, lease_id, request_params)
                SELECT :key, lease_id, :params
                FROM new_lease
            )
            SELECT * FROM new_lease
            """
        ),
        {
            "antenna_id": antenna_id,
            "controller": controller,
            "duration": duration_seconds,
            "key": idempotency_key,
            "params": fingerprint,
        },
    ).mappings().one()
    return {**dict(row), "replay": False}


def get_lease_by_token(conn: Connection, token: str) -> dict[str, Any] | None:
    row = conn.execute(
        text(
            """
            SELECT id AS lease_id, antenna_id, controller,
                   token, acquired_at, expires_at, released_at,
                   last_command_sequence, last_progress_at,
                   (released_at IS NULL AND expires_at > clock_timestamp())
                       AS active
            FROM leases
            WHERE token = :token
            """
        ),
        {"token": token},
    ).mappings().first()
    return dict(row) if row is not None else None


def report_progress(conn: Connection, token: str, sequence: int) -> dict[str, Any]:
    """Confirm that the current lease holder has executed command ``sequence``.

    All work happens in one READ COMMITTED transaction:

    1. The token must identify an existing lease; unknown tokens raise
       ``LEASE_NOT_FOUND`` before any lock or write.
    2. ``SELECT ... FROM antennas WHERE id = :antenna_id FOR UPDATE`` locks
       the lease's antenna row, so concurrent reports (and acquisitions) for
       the same antenna serialise. The lease is then re-read *after* the
       lock, so the current high-water mark is always the committed one.
    3. The lease must still satisfy ``expires_at > clock_timestamp()``:
       expired tokens raise ``LEASE_EXPIRED``. Reporting never extends a
       lease. Neither error path issues a write.
    4. Sequence numbers may only advance. An equal sequence is a replay: the
       originally recorded sequence/time are returned byte-stably. A smaller
       sequence raises ``PROGRESS_REGRESSION`` and changes nothing.
    5. Advancing updates ``last_command_sequence`` and stamps
       ``last_progress_at = clock_timestamp()`` (database clock only).
    """
    # Defence in depth alongside the Pydantic boundary check.
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise APIError(
            422,
            "VALIDATION_ERROR",
            "sequence 必须为非负整数。",
            {"sequence": sequence},
        )

    # 1. Resolve the lease and lock its antenna. Locking the antenna row
    #    (rather than the lease row) matches the acquisition lock order and
    #    serialises progress reports with expiry hand-over for that antenna.
    found = conn.execute(
        text("SELECT id, antenna_id FROM leases WHERE token = :token"),
        {"token": token},
    ).mappings().first()
    if found is None:
        raise APIError(
            404,
            "LEASE_NOT_FOUND",
            "未知租约令牌。",
            {"lease_token": token},
        )

    conn.execute(
        text("SELECT id FROM antennas WHERE id = :antenna_id FOR UPDATE"),
        {"antenna_id": found.antenna_id},
    )

    # Re-read the lease AFTER the antenna lock: under READ COMMITTED a
    # concurrent report that held the lock has now committed, so the
    # high-water mark seen here is current. This is what makes regressions
    # impossible under contention and leaves the maximum sequence as the
    # final value.
    lease = conn.execute(
        text(
            """
            SELECT id AS lease_id, antenna_id, expires_at, released_at,
                   last_command_sequence, last_progress_at
            FROM leases
            WHERE id = :lease_id
            """
        ),
        {"lease_id": found.id},
    ).mappings().one()

    # 2. Expiry is evaluated against the database clock: a report never
    #    extends the lease and never lands on an expired one.
    now = conn.execute(text("SELECT clock_timestamp()")).scalar_one()
    if lease.released_at is not None or lease.expires_at <= now:
        raise APIError(
            409,
            "LEASE_EXPIRED",
            "租约已到期或已提前释放，不能再上报指令进度。",
            {
                "lease_token": token,
                "antenna_id": lease.antenna_id,
                "expires_at": lease.expires_at.isoformat(),
            },
        )

    # 3. Equal sequence -> idempotent replay of the original record; smaller
    #    sequence -> stable regression error. No write in either case.
    if (
        lease.last_command_sequence is not None
        and lease.last_command_sequence >= sequence
    ):
        if lease.last_command_sequence == sequence:
            return {
                "lease_token": token,
                "last_command_sequence": lease.last_command_sequence,
                "last_progress_at": lease.last_progress_at,
            }
        raise APIError(
            409,
            "PROGRESS_REGRESSION",
            "指令序号只能递增，不能回退到更小的序号。",
            {
                "lease_token": token,
                "reported_sequence": sequence,
                "last_command_sequence": lease.last_command_sequence,
            },
        )

    # 4. Advance the high-water mark; the recording timestamp is generated by
    #    the database clock inside the same UPDATE.
    row = conn.execute(
        text(
            """
            UPDATE leases
            SET last_command_sequence = :sequence,
                last_progress_at = clock_timestamp()
            WHERE id = :lease_id
            RETURNING last_command_sequence, last_progress_at
            """
        ),
        {"sequence": sequence, "lease_id": lease.lease_id},
    ).mappings().one()
    return {
        "lease_token": token,
        "last_command_sequence": row.last_command_sequence,
        "last_progress_at": row.last_progress_at,
    }


def release_lease(conn: Connection, token: str) -> dict[str, Any]:
    """Release an active lease early under the antenna's row lock."""
    found = conn.execute(
        text("SELECT id, antenna_id FROM leases WHERE token = :token"),
        {"token": token},
    ).mappings().first()
    if found is None:
        raise APIError(
            404,
            "LEASE_NOT_FOUND",
            "未知租约令牌。",
            {"lease_token": token},
        )

    conn.execute(
        text("SELECT id FROM antennas WHERE id = :antenna_id FOR UPDATE"),
        {"antenna_id": found.antenna_id},
    ).one()
    lease = conn.execute(
        text(
            """
            SELECT id AS lease_id, antenna_id, controller, token,
                   acquired_at, expires_at, released_at,
                   last_command_sequence, last_progress_at,
                   (expires_at > clock_timestamp()) AS unexpired
            FROM leases
            WHERE id = :lease_id
            """
        ),
        {"lease_id": found.id},
    ).mappings().one()

    if lease.released_at is not None:
        return _release_result(lease, lease.released_at)
    if not lease.unexpired:
        raise APIError(
            409,
            "LEASE_EXPIRED",
            "租约已自然到期，无需释放。",
            {
                "lease_token": token,
                "expires_at": lease.expires_at.isoformat(),
            },
        )

    released_at = conn.execute(
        text(
            """
            UPDATE leases
            SET released_at = clock_timestamp()
            WHERE id = :lease_id
            RETURNING released_at
            """
        ),
        {"lease_id": lease.lease_id},
    ).scalar_one()
    return _release_result(lease, released_at)


def _release_result(row: Any, released_at: Any) -> dict[str, Any]:
    result = dict(row)
    result.pop("unexpired", None)
    result["released_at"] = released_at
    result["active"] = False
    return result


def renew_lease(
    conn: Connection,
    token: str,
    *,
    additional_seconds: int,
    idempotency_key: str,
) -> dict[str, Any]:
    """Extend an active lease by ``additional_seconds`` from its current expiry.

    All work happens in one READ COMMITTED transaction, mirroring the
    acquisition flow:

    1. ``pg_advisory_xact_lock(hashtext(:key))`` serialises retries that share
       the renewal idempotency key, so a lost response can never extend the
       lease twice.
    2. Look up the stored renewal record. Same parameters -> replay the
       recorded before/after expiry byte-stably; different parameters (other
       token or other seconds) -> stable ``IDEMPOTENCY_CONFLICT``. Neither
       path writes.
    3. The token must identify an existing lease; unknown tokens raise
       ``LEASE_NOT_FOUND`` before any lock or write.
    4. ``SELECT ... FROM antennas WHERE id = :antenna_id FOR UPDATE`` — the
       same antenna row lock acquisition takes. Renewal and expiry hand-over
       are therefore serialised: exactly one of them wins at the boundary.
    5. Re-read the lease AFTER the lock and confirm
       ``released_at IS NULL AND expires_at > clock_timestamp()`` (database
       clock only). An expired or released lease raises ``LEASE_EXPIRED`` and
       is left untouched.
    6. Extend from the ORIGINAL expiry — ``expires_at = expires_at +
       make_interval(...)``, never from "now" — and record the renewal
       (before/after expiry copied from the same UPDATE) atomically.
    """
    # Defence in depth alongside the Pydantic boundary check.
    if not (
        isinstance(additional_seconds, int)
        and not isinstance(additional_seconds, bool)
        and MIN_RENEW_SECONDS <= additional_seconds <= MAX_RENEW_SECONDS
    ):
        raise APIError(
            422,
            "VALIDATION_ERROR",
            f"追加秒数必须为 {MIN_RENEW_SECONDS} 至 {MAX_RENEW_SECONDS} 秒之间的整数。",
            {
                "additional_seconds": additional_seconds,
                "min": MIN_RENEW_SECONDS,
                "max": MAX_RENEW_SECONDS,
            },
        )

    fingerprint = _canonical_renew_params(token, additional_seconds)

    # 1. Serialise transactions sharing one renewal idempotency key.
    conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": idempotency_key},
    )

    # 2. Replay or stable conflict. The recorded before/after expiry comes
    #    back byte-identically; only the replay flag differs.
    existing = conn.execute(
        text(
            """
            SELECT r.lease_id, r.request_params,
                   r.previous_expires_at, r.new_expires_at,
                   l.token AS lease_token
            FROM lease_renewals r
            JOIN leases l ON l.id = r.lease_id
            WHERE r.idempotency_key = :key
            """
        ),
        {"key": idempotency_key},
    ).mappings().first()

    if existing is not None:
        if existing.request_params != fingerprint:
            raise APIError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "同一幂等键曾用于不同的续期参数，拒绝执行。",
                {
                    "idempotency_key": idempotency_key,
                    "original_params": existing.request_params,
                    "request_params": fingerprint,
                },
            )
        return {
            "lease_token": existing.lease_token,
            "previous_expires_at": existing.previous_expires_at,
            "expires_at": existing.new_expires_at,
            "replay": True,
        }

    # 3. Resolve the lease (no writes before this point for unknown tokens).
    found = conn.execute(
        text("SELECT id, antenna_id FROM leases WHERE token = :token"),
        {"token": token},
    ).mappings().first()
    if found is None:
        raise APIError(
            404,
            "LEASE_NOT_FOUND",
            "未知租约令牌。",
            {"lease_token": token},
        )

    # 4. Lock the antenna row: the same lock acquisition, progress and
    #    release take, so a renewal can never overlap an expiry hand-over.
    conn.execute(
        text("SELECT id FROM antennas WHERE id = :antenna_id FOR UPDATE"),
        {"antenna_id": found.antenna_id},
    )

    # 5. Re-read after the lock and confirm the lease is still held, judged
    #    purely by the database clock. The boundary belongs to the incoming
    #    acquirer: expires_at <= clock_timestamp() is already handed over.
    lease = conn.execute(
        text(
            """
            SELECT id AS lease_id, antenna_id, expires_at, released_at,
                   (released_at IS NULL AND expires_at > clock_timestamp())
                       AS still_held
            FROM leases
            WHERE id = :lease_id
            """
        ),
        {"lease_id": found.id},
    ).mappings().one()

    if not lease.still_held:
        raise APIError(
            409,
            "LEASE_EXPIRED",
            "租约已到期或已提前释放，不能续期。",
            {
                "lease_token": token,
                "antenna_id": lease.antenna_id,
                "expires_at": lease.expires_at.isoformat(),
            },
        )

    # 6. Extend from the lease's own current expiry and record the renewal in
    #    the same statement pair of one transaction. ``previous_expires_at``
    #    is recovered from the UPDATE itself, so the record can never disagree
    #    with the lease row.
    row = conn.execute(
        text(
            """
            WITH updated AS (
                UPDATE leases
                SET expires_at = expires_at + make_interval(secs => :additional)
                WHERE id = :lease_id
                RETURNING id AS lease_id,
                          expires_at - make_interval(secs => :additional)
                              AS previous_expires_at,
                          expires_at AS new_expires_at
            ), recorded AS (
                INSERT INTO lease_renewals
                    (lease_id, idempotency_key, request_params,
                     additional_seconds, previous_expires_at, new_expires_at)
                SELECT lease_id, :key, :params, :additional,
                       previous_expires_at, new_expires_at
                FROM updated
            )
            SELECT previous_expires_at, new_expires_at FROM updated
            """
        ),
        {
            "lease_id": lease.lease_id,
            "additional": additional_seconds,
            "key": idempotency_key,
            "params": fingerprint,
        },
    ).mappings().one()
    return {
        "lease_token": token,
        "previous_expires_at": row.previous_expires_at,
        "expires_at": row.new_expires_at,
        "replay": False,
    }
