"""Lease renewal: extend a live pass window without releasing control.

``POST /leases/{lease_token}/renew`` appends 5–120 seconds to the CURRENT
expiry of a still-held lease (never to "now"), inside one transaction that
takes the same antenna row lock as acquisition. A renewal and an expiry
handover therefore serialise and exactly one of them succeeds at the
boundary. Each renewal carries an idempotency key: the first success writes
a renewal record, a same-key same-params retry replays it byte-stably
(``replay: true``) and never extends twice, and a same-key reuse with
different parameters is a stable ``IDEMPOTENCY_CONFLICT``.

Everything runs against the real API + real PostgreSQL; no mocks.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from conftest import (
    KNOWN_ANTENNA,
    acquire,
    active_lease_count,
    count_rows,
    insert_expired_lease,
    make_key,
    release,
    total_lease_count,
)

# ISO-8601 with an explicit UTC offset (``+00:00``, never the bare "Z").
ISO_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}$"
)

# The renewal feature must not change the pre-existing response shapes.
ACQUIRE_RESPONSE_FIELDS = {
    "antenna_id",
    "controller",
    "lease_token",
    "acquired_at",
    "expires_at",
    "replay",
}
STATUS_RESPONSE_FIELDS = {
    "antenna_id",
    "controller",
    "lease_token",
    "acquired_at",
    "expires_at",
    "active",
    "last_command_sequence",
    "last_progress_at",
    "released_at",
}
RENEW_RESPONSE_FIELDS = {
    "lease_token",
    "expires_before",
    "expires_after",
    "replay",
}


def renew(
    client: httpx.Client,
    token: str,
    additional_seconds: int,
    key: str | None = None,
) -> httpx.Response:
    return client.post(
        f"/leases/{token}/renew",
        json={
            "additional_seconds": additional_seconds,
            "idempotency_key": key or make_key("renew"),
        },
    )


def _lease_row(db_engine: Engine, token: str) -> dict:
    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT token, controller, acquired_at, expires_at, released_at,
                       last_command_sequence, last_progress_at
                FROM leases WHERE token = :token
                """
            ),
            {"token": token},
        ).mappings().one()
    return dict(row)


def _renewal_rows(db_engine: Engine, token: str) -> list[dict]:
    with db_engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT r.idempotency_key, r.additional_seconds,
                       r.expires_before, r.expires_after
                FROM lease_renewals r
                JOIN leases l ON l.id = r.lease_id
                WHERE l.token = :token
                ORDER BY r.id
                """
            ),
            {"token": token},
        ).mappings().all()
    return [dict(r) for r in rows]


def _wait_past(deadline: datetime, margin: float = 0.3) -> None:
    while datetime.now(timezone.utc) <= deadline:
        time.sleep(0.05)
    time.sleep(margin)  # boundary margin


def test_renew_extends_from_the_original_expiry(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-01", duration_seconds=30)
    # The renewal feature leaves the acquisition response shape untouched.
    assert set(held.json()) == ACQUIRE_RESPONSE_FIELDS
    token = held.json()["lease_token"]
    original_expiry = held.json()["expires_at"]

    resp = renew(http_client, token, 15)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == RENEW_RESPONSE_FIELDS
    assert body["lease_token"] == token
    assert body["replay"] is False
    # Both boundaries are reported, in the same explicit-offset format as
    # every other timestamp in the API.
    assert body["expires_before"] == original_expiry
    assert ISO_OFFSET.match(body["expires_before"])
    assert ISO_OFFSET.match(body["expires_after"])
    # The extension grows from the ORIGINAL expiry, never from "now".
    assert datetime.fromisoformat(body["expires_after"]) == (
        datetime.fromisoformat(original_expiry) + timedelta(seconds=15)
    )

    # The status query reflects the new boundary byte-identically and keeps
    # its original shape (no fields added or removed).
    status = http_client.get(f"/leases/{token}")
    assert set(status.json()) == STATUS_RESPONSE_FIELDS
    assert status.json()["expires_at"] == body["expires_after"]
    assert status.json()["active"] is True

    # The committed lease row and the renewal record agree with the response.
    row = _lease_row(db_engine, token)
    assert row["expires_at"].isoformat() == body["expires_after"]
    renewals = _renewal_rows(db_engine, token)
    assert len(renewals) == 1
    record = renewals[0]
    assert record["additional_seconds"] == 15
    assert record["expires_before"].isoformat() == original_expiry
    assert record["expires_after"].isoformat() == body["expires_after"]


def test_renewed_lease_hands_over_at_the_new_boundary(http_client, db_engine):
    held = acquire(http_client, antenna_id=KNOWN_ANTENNA, duration_seconds=5)
    token = held.json()["lease_token"]
    old_expiry = held.json()["expires_at"]

    renewed = renew(http_client, token, 10)
    assert renewed.status_code == 200
    new_expiry = renewed.json()["expires_after"]

    # Past the ORIGINAL boundary the antenna is still held: a contender is
    # rejected and told to wait for the EXTENDED expiry.
    _wait_past(datetime.fromisoformat(old_expiry))
    contender = acquire(
        http_client,
        antenna_id=KNOWN_ANTENNA,
        controller="early-bird",
        duration_seconds=5,
    )
    assert contender.status_code == 409
    error = contender.json()["error"]
    assert error["code"] == "ANTENNA_BUSY"
    assert error["details"]["held_by_lease"] == token
    assert error["details"]["expires_at"] == new_expiry
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1

    # Only at the NEW boundary does the handover happen.
    _wait_past(datetime.fromisoformat(new_expiry))
    successor = acquire(
        http_client,
        antenna_id=KNOWN_ANTENNA,
        controller="successor",
        duration_seconds=10,
    )
    assert successor.status_code == 200
    assert successor.json()["lease_token"] != token
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1
    assert total_lease_count(db_engine, KNOWN_ANTENNA) == 2

    # The old token can no longer be renewed after the handover.
    late = renew(http_client, token, 10)
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "LEASE_EXPIRED"
    assert _lease_row(db_engine, token)["expires_at"].isoformat() == new_expiry


def test_same_key_retry_replays_and_extends_only_once(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-02", duration_seconds=30)
    token = held.json()["lease_token"]
    key = make_key("renew-replay")

    first = renew(http_client, token, 10, key=key)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["replay"] is False

    time.sleep(0.1)  # a second extension would move expires_after

    for _ in range(3):
        again = renew(http_client, token, 10, key=key)
        assert again.status_code == 200
        body = again.json()
        assert body["replay"] is True
        # Business fields are byte-identical to the first result.
        assert body["lease_token"] == token
        assert body["expires_before"] == first_body["expires_before"]
        assert body["expires_after"] == first_body["expires_after"]

    # The lease grew exactly once: original + 10s, never +20s.
    expected = datetime.fromisoformat(held.json()["expires_at"]) + timedelta(
        seconds=10
    )
    assert datetime.fromisoformat(first_body["expires_after"]) == expected
    status = http_client.get(f"/leases/{token}").json()
    assert status["expires_at"] == first_body["expires_after"]
    assert _lease_row(db_engine, token)["expires_at"] == expected
    assert len(_renewal_rows(db_engine, token)) == 1


def test_concurrent_same_key_renews_extend_exactly_once(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-03", duration_seconds=60)
    token = held.json()["lease_token"]
    original = datetime.fromisoformat(held.json()["expires_at"])
    key = make_key("renew-same")

    workers = 6
    barrier = threading.Barrier(workers)

    def one(_: int) -> httpx.Response:
        barrier.wait(timeout=10)
        return renew(http_client, token, 10, key=key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(one, range(workers)))

    assert all(r.status_code == 200 for r in responses), [
        r.text for r in responses
    ]
    bodies = [r.json() for r in responses]
    # Exactly one first-time success; every other call replays it.
    assert sum(1 for b in bodies if b["replay"] is False) == 1
    assert sum(1 for b in bodies if b["replay"] is True) == workers - 1
    # Every caller observed the identical before/after pair.
    assert len({b["expires_before"] for b in bodies}) == 1
    assert len({b["expires_after"] for b in bodies}) == 1
    extended = datetime.fromisoformat(bodies[0]["expires_after"])
    assert extended == original + timedelta(seconds=10)
    # The lease grew exactly once no matter how many retries raced.
    assert _lease_row(db_engine, token)["expires_at"] == extended
    assert len(_renewal_rows(db_engine, token)) == 1


def test_distinct_keys_stack_and_each_key_replays_its_own_record(
    http_client, db_engine
):
    held = acquire(http_client, antenna_id="ANT-04", duration_seconds=30)
    token = held.json()["lease_token"]
    original = datetime.fromisoformat(held.json()["expires_at"])

    key_a, key_b = make_key("renew-a"), make_key("renew-b")
    first = renew(http_client, token, 10, key=key_a)
    assert first.status_code == 200 and first.json()["replay"] is False
    second = renew(http_client, token, 5, key=key_b)
    assert second.status_code == 200 and second.json()["replay"] is False

    # Extensions accumulate from the previous expiry, never from "now".
    first_after = datetime.fromisoformat(first.json()["expires_after"])
    assert first_after == original + timedelta(seconds=10)
    assert datetime.fromisoformat(second.json()["expires_before"]) == first_after
    second_after = datetime.fromisoformat(second.json()["expires_after"])
    assert second_after == original + timedelta(seconds=15)

    status = http_client.get(f"/leases/{token}").json()
    assert status["expires_at"] == second.json()["expires_after"]
    assert _lease_row(db_engine, token)["expires_at"] == second_after
    assert len(_renewal_rows(db_engine, token)) == 2

    # Replaying key A returns ITS OWN recorded result, not the latest state,
    # and changes nothing.
    replay_a = renew(http_client, token, 10, key=key_a)
    assert replay_a.status_code == 200
    assert replay_a.json()["replay"] is True
    assert replay_a.json()["expires_after"] == first.json()["expires_after"]
    assert _lease_row(db_engine, token)["expires_at"] == second_after
    assert len(_renewal_rows(db_engine, token)) == 2


def test_concurrent_distinct_key_renews_all_stack(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-05", duration_seconds=60)
    token = held.json()["lease_token"]
    original = datetime.fromisoformat(held.json()["expires_at"])

    workers = 5
    keys = [make_key(f"renew-{i}") for i in range(workers)]
    barrier = threading.Barrier(workers)

    def one(key: str) -> httpx.Response:
        barrier.wait(timeout=10)
        return renew(http_client, token, 5, key=key)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        responses = list(pool.map(one, keys))

    assert all(r.status_code == 200 for r in responses), [
        r.text for r in responses
    ]
    assert all(r.json()["replay"] is False for r in responses)
    # Five +5s extensions applied in some serial order: distinct results,
    # final expiry is original + 25s.
    afters = {r.json()["expires_after"] for r in responses}
    assert len(afters) == workers
    final_expiry = original + timedelta(seconds=5 * workers)
    assert max(datetime.fromisoformat(a) for a in afters) == final_expiry
    assert _lease_row(db_engine, token)["expires_at"] == final_expiry
    assert len(_renewal_rows(db_engine, token)) == workers


def test_same_key_with_different_params_conflicts_and_writes_nothing(
    http_client, db_engine
):
    held = acquire(http_client, antenna_id="ANT-06", duration_seconds=30)
    token = held.json()["lease_token"]
    key = make_key("renew-conflict")

    first = renew(http_client, token, 10, key=key)
    assert first.status_code == 200
    extended = first.json()["expires_after"]

    # Same key, different duration -> stable conflict, stable on repeat.
    for bad_seconds in (11, 120, 11):
        conflict = renew(http_client, token, bad_seconds, key=key)
        assert conflict.status_code == 409
        error = conflict.json()["error"]
        assert error["code"] == "IDEMPOTENCY_CONFLICT"
        assert error["details"]["idempotency_key"] == key

    # Same key, different token -> also a conflict (the token is a parameter).
    other = acquire(http_client, antenna_id="ANT-01", duration_seconds=30)
    other_token = other.json()["lease_token"]
    cross = renew(http_client, other_token, 10, key=key)
    assert cross.status_code == 409
    assert cross.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

    # Nothing was rewritten: both leases keep their expiry and the only
    # renewal record is the first one.
    assert _lease_row(db_engine, token)["expires_at"].isoformat() == extended
    assert http_client.get(f"/leases/{token}").json()["expires_at"] == extended
    assert (
        _lease_row(db_engine, other_token)["expires_at"].isoformat()
        == other.json()["expires_at"]
    )
    assert len(_renewal_rows(db_engine, token)) == 1
    assert len(_renewal_rows(db_engine, other_token)) == 0

    # The original key still replays its own result.
    replay = renew(http_client, token, 10, key=key)
    assert replay.status_code == 200
    assert replay.json()["replay"] is True
    assert replay.json()["expires_after"] == extended


def test_unknown_token_is_lease_not_found_and_consumes_nothing(
    http_client, db_engine
):
    key = make_key("renew-unknown")
    leases_before = count_rows(db_engine, "SELECT count(*) FROM leases")

    resp = renew(http_client, f"no-such-{uuid.uuid4()}", 10, key=key)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "LEASE_NOT_FOUND"

    assert count_rows(db_engine, "SELECT count(*) FROM leases") == leases_before
    assert count_rows(db_engine, "SELECT count(*) FROM lease_renewals") == 0

    # The rejection stored nothing: the same key is still usable for a real
    # renewal afterwards.
    held = acquire(http_client, antenna_id="ANT-02", duration_seconds=30)
    ok = renew(http_client, held.json()["lease_token"], 10, key=key)
    assert ok.status_code == 200
    assert ok.json()["replay"] is False


def test_expired_token_is_lease_expired_and_data_unchanged(
    http_client, db_engine
):
    planted = insert_expired_lease(
        db_engine, antenna_id="ANT-03", age_seconds=2, ttl_seconds=10
    )
    token = planted["token"]
    before = _lease_row(db_engine, token)

    for _ in range(2):
        resp = renew(http_client, token, 10)
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "LEASE_EXPIRED"
        assert error["details"]["lease_token"] == token
        assert error["details"]["expires_at"] == before["expires_at"].isoformat()
        # The rejection is stable and rewrites nothing.
        assert _lease_row(db_engine, token) == before
        assert len(_renewal_rows(db_engine, token)) == 0


def test_renew_of_released_lease_is_rejected(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-04", duration_seconds=60)
    token = held.json()["lease_token"]
    assert release(http_client, token).status_code == 200
    before = _lease_row(db_engine, token)

    resp = renew(http_client, token, 10)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "LEASE_EXPIRED"
    assert _lease_row(db_engine, token) == before
    assert len(_renewal_rows(db_engine, token)) == 0


def test_rejected_renew_does_not_affect_the_later_holder(
    http_client, db_engine
):
    first = acquire(http_client, antenna_id="ANT-05", duration_seconds=5)
    old_token = first.json()["lease_token"]

    # Wait out the original lease (database-computed boundary), then let a
    # successor take over.
    _wait_past(datetime.fromisoformat(first.json()["expires_at"]))
    successor = acquire(
        http_client,
        antenna_id="ANT-05",
        controller="next-shift",
        duration_seconds=60,
    )
    assert successor.status_code == 200
    new_token = successor.json()["lease_token"]
    new_expiry = successor.json()["expires_at"]

    # Renewing the OLD token fails and must not touch the new holder.
    resp = renew(http_client, old_token, 30)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "LEASE_EXPIRED"

    status = http_client.get(f"/leases/{new_token}").json()
    assert status["active"] is True
    assert status["expires_at"] == new_expiry
    assert _lease_row(db_engine, new_token)["expires_at"].isoformat() == new_expiry
    assert active_lease_count(db_engine, "ANT-05") == 1
    assert count_rows(db_engine, "SELECT count(*) FROM lease_renewals") == 0


def test_concurrent_renew_and_contenders_while_active(http_client, db_engine):
    held = acquire(http_client, antenna_id="ANT-06", duration_seconds=60)
    token = held.json()["lease_token"]
    original_expiry = held.json()["expires_at"]

    # One renewal races three would-be acquirers, all released from a barrier.
    kinds = ["renew", "acquire", "acquire", "acquire"]
    barrier = threading.Barrier(len(kinds))

    def run(kind: str) -> httpx.Response:
        barrier.wait(timeout=10)
        if kind == "renew":
            return renew(http_client, token, 30)
        return http_client.post(
            "/leases",
            json={
                "antenna_id": "ANT-06",
                "controller": f"contender-{make_key()}",
                "duration_seconds": 30,
                "idempotency_key": make_key(),
            },
        )

    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        responses = list(pool.map(run, kinds))

    renewed = responses[0]
    assert renewed.status_code == 200, renewed.text
    body = renewed.json()
    assert body["replay"] is False
    assert body["expires_before"] == original_expiry

    # Every contender lost; depending on who took the antenna lock first it
    # saw either the original or the already-extended expiry — never a grant.
    for resp in responses[1:]:
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "ANTENNA_BUSY"
        assert error["details"]["held_by_lease"] == token
        assert error["details"]["expires_at"] in (
            original_expiry,
            body["expires_after"],
        )

    assert active_lease_count(db_engine, "ANT-06") == 1
    assert total_lease_count(db_engine, "ANT-06") == 1
    assert len(_renewal_rows(db_engine, token)) == 1
    assert (
        http_client.get(f"/leases/{token}").json()["expires_at"]
        == body["expires_after"]
    )


def test_concurrent_renew_and_handover_never_double_controls(
    http_client, db_engine
):
    """A renewal and an expiry handover race at the boundary: exactly one of
    them succeeds, and the antenna never ends up doubly controlled."""
    antennas = ["ANT-01", "ANT-02", "ANT-03", "ANT-04", "ANT-05", "ANT-06"]
    for round_no, antenna in enumerate(antennas):
        # Alternate the two sides of the boundary: a lease expiring 0.35s in
        # the future (either side may win the race) and a lease that expired
        # 0.35s ago (the handover must win deterministically).
        age_seconds = -0.35 if round_no % 2 == 0 else 0.35
        planted = insert_expired_lease(
            db_engine, antenna_id=antenna, age_seconds=age_seconds, ttl_seconds=10
        )
        token = planted["token"]

        barrier = threading.Barrier(2)

        def do_renew() -> httpx.Response:
            barrier.wait(timeout=10)
            return renew(http_client, token, 30)

        def do_acquire() -> httpx.Response:
            barrier.wait(timeout=10)
            return http_client.post(
                "/leases",
                json={
                    "antenna_id": antenna,
                    "controller": "boundary-contender",
                    "duration_seconds": 30,
                    "idempotency_key": make_key("boundary"),
                },
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            renew_future = pool.submit(do_renew)
            acquire_future = pool.submit(do_acquire)
            renew_resp = renew_future.result()
            acquire_resp = acquire_future.result()

        renew_ok = renew_resp.status_code == 200
        acquire_ok = acquire_resp.status_code == 200
        # Exactly one side wins; never both, never neither.
        assert renew_ok != acquire_ok, (renew_resp.text, acquire_resp.text)

        if renew_ok:
            assert acquire_resp.status_code == 409
            assert acquire_resp.json()["error"]["code"] == "ANTENNA_BUSY"
            body = renew_resp.json()
            assert body["replay"] is False
            status = http_client.get(f"/leases/{token}").json()
            assert status["active"] is True
            assert status["expires_at"] == body["expires_after"]
            assert total_lease_count(db_engine, antenna) == 1
            assert len(_renewal_rows(db_engine, token)) == 1
        else:
            assert renew_resp.status_code == 409
            assert renew_resp.json()["error"]["code"] == "LEASE_EXPIRED"
            assert acquire_resp.status_code == 200
            new_token = acquire_resp.json()["lease_token"]
            assert new_token != token
            assert http_client.get(f"/leases/{token}").json()["active"] is False
            # The rejected renewal did not rewrite the expired lease.
            assert (
                _lease_row(db_engine, token)["expires_at"]
                == planted["expires_at"]
            )
            assert total_lease_count(db_engine, antenna) == 2
            assert len(_renewal_rows(db_engine, token)) == 0

        # Whatever the outcome: exactly one active lease, no double control.
        assert active_lease_count(db_engine, antenna) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"additional_seconds": 4, "idempotency_key": "k"},
        {"additional_seconds": 121, "idempotency_key": "k"},
        {"additional_seconds": "30", "idempotency_key": "k"},
        {"additional_seconds": 10.5, "idempotency_key": "k"},
        {"additional_seconds": True, "idempotency_key": "k"},
        {"additional_seconds": 10},  # missing idempotency key
        {"idempotency_key": "k"},  # missing additional_seconds
        {"additional_seconds": 10, "idempotency_key": "   "},
        {"additional_seconds": 10, "idempotency_key": "k", "extra": 1},
    ],
)
def test_invalid_renew_payloads_are_422_and_write_nothing(
    http_client, db_engine, payload
):
    held = acquire(http_client, antenna_id="ANT-01", duration_seconds=30)
    token = held.json()["lease_token"]
    before = _lease_row(db_engine, token)

    resp = http_client.post(f"/leases/{token}/renew", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    assert _lease_row(db_engine, token) == before
    assert count_rows(db_engine, "SELECT count(*) FROM lease_renewals") == 0


def test_boundary_additional_seconds_5_and_120_are_accepted(http_client):
    held = acquire(http_client, antenna_id="ANT-02", duration_seconds=30)
    token = held.json()["lease_token"]
    original = datetime.fromisoformat(held.json()["expires_at"])

    low = renew(http_client, token, 5)
    assert low.status_code == 200
    assert datetime.fromisoformat(low.json()["expires_after"]) == (
        original + timedelta(seconds=5)
    )

    high = renew(http_client, token, 120)
    assert high.status_code == 200
    assert datetime.fromisoformat(high.json()["expires_after"]) == (
        original + timedelta(seconds=125)
    )
