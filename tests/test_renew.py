"""Lease renewal: the current holder extends its expiry while the pass
window grows, without releasing control and re-contending.

``POST /leases/{lease_token}/renew`` adds 5–120 seconds on top of the
lease's CURRENT expiry (never "now + seconds"), inside the same antenna row
lock acquisition uses, so a renewal and an expiry hand-over cannot both win.
The renewal is recorded (``lease_renewals``) under the caller's idempotency
key: a same-key/same-params retry replays the recorded before/after expiry
and never extends twice; the same key with different parameters is a stable
conflict. Everything runs against the real API + PostgreSQL; no mocks.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
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
    renew,
)

# ISO-8601 with an explicit UTC offset (same contract as every other
# timestamp the service emits: ``+00:00``, never the bare "Z" shorthand).
ISO_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}$"
)

RENEW_RESPONSE_FIELDS = {
    "lease_token",
    "previous_expires_at",
    "expires_at",
    "replay",
}

# The renew feature must not change the existing response shapes.
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


def _lease_row(db_engine: Engine, token: str) -> dict:
    with db_engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT token, acquired_at, expires_at, released_at
                FROM leases WHERE token = :token
                """
            ),
            {"token": token},
        ).mappings().one()
    return dict(row)


def _renewal_count(db_engine: Engine, key: str | None = None) -> int:
    if key is None:
        return count_rows(db_engine, "SELECT count(*) FROM lease_renewals")
    return count_rows(
        db_engine,
        "SELECT count(*) FROM lease_renewals WHERE idempotency_key = :k",
        k=key,
    )


def test_renew_extends_and_hands_over_at_the_new_boundary(http_client, db_engine):
    granted = acquire(http_client, antenna_id=KNOWN_ANTENNA, duration_seconds=5)
    assert granted.status_code == 200
    token = granted.json()["lease_token"]
    original_expiry = granted.json()["expires_at"]

    renewed = renew(http_client, token, additional_seconds=5)
    assert renewed.status_code == 200, renewed.text
    body = renewed.json()
    assert set(body) == RENEW_RESPONSE_FIELDS
    assert body["replay"] is False
    assert body["lease_token"] == token
    # Before/after expiry are reported, and the extension accumulates from
    # the ORIGINAL expiry, not from the renewal moment.
    assert body["previous_expires_at"] == original_expiry
    assert datetime.fromisoformat(body["expires_at"]) == datetime.fromisoformat(
        original_expiry
    ) + timedelta(seconds=5)
    for field in ("previous_expires_at", "expires_at"):
        assert ISO_OFFSET.match(body[field]), body[field]

    # The database row and the status query agree byte-for-byte with the
    # renew response.
    assert _lease_row(db_engine, token)["expires_at"].isoformat() == body[
        "expires_at"
    ]
    status = http_client.get(f"/leases/{token}")
    assert set(status.json()) == STATUS_RESPONSE_FIELDS
    assert status.json()["expires_at"] == body["expires_at"]
    assert status.json()["active"] is True

    # A contender is still refused, and the busy details advertise the NEW
    # hand-over time.
    contender = acquire(http_client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)
    assert contender.status_code == 409
    busy = contender.json()["error"]
    assert busy["code"] == "ANTENNA_BUSY"
    assert busy["details"]["held_by_lease"] == token
    assert busy["details"]["expires_at"] == body["expires_at"]

    # Past the ORIGINAL boundary the renewed lease is still in force: the
    # extension genuinely moved the hand-over.
    original_deadline = datetime.fromisoformat(original_expiry)
    while datetime.now(timezone.utc) <= original_deadline:
        time.sleep(0.05)
    time.sleep(0.2)
    assert http_client.get(f"/leases/{token}").json()["active"] is True
    still_busy = acquire(http_client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)
    assert still_busy.status_code == 409
    assert still_busy.json()["error"]["code"] == "ANTENNA_BUSY"

    # Past the NEW boundary the antenna finally hands over.
    new_deadline = datetime.fromisoformat(body["expires_at"])
    while datetime.now(timezone.utc) <= new_deadline:
        time.sleep(0.05)
    time.sleep(0.3)  # boundary margin
    assert http_client.get(f"/leases/{token}").json()["active"] is False

    successor = acquire(
        http_client, antenna_id=KNOWN_ANTENNA, controller="successor",
        duration_seconds=30,
    )
    assert successor.status_code == 200, successor.text
    assert successor.json()["lease_token"] != token
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1
    assert count_rows(db_engine, "SELECT count(*) FROM leases") == 2

    # A fresh renewal attempt (new key) on the handed-over lease is refused;
    # the recorded key's replay after expiry has its own dedicated test below.
    late = renew(http_client, token, additional_seconds=5)
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "LEASE_EXPIRED"
    assert _renewal_count(db_engine) == 1


def test_same_key_retry_extends_only_once_and_replays(http_client, db_engine):
    token = acquire(http_client, antenna_id="ANT-02", duration_seconds=30).json()[
        "lease_token"
    ]
    key = make_key("renew")

    first = renew(http_client, token, additional_seconds=10, idempotency_key=key)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["replay"] is False

    time.sleep(0.05)  # a second extension would move the DB-clock value

    for _ in range(3):
        again = renew(
            http_client, token, additional_seconds=10, idempotency_key=key
        )
        assert again.status_code == 200
        body = again.json()
        assert body["replay"] is True
        # Every business field is identical to the first result; only the
        # replay flag differs.
        assert body["lease_token"] == first_body["lease_token"]
        assert body["previous_expires_at"] == first_body["previous_expires_at"]
        assert body["expires_at"] == first_body["expires_at"]

    # Extended exactly once: one renewal record, and the lease expiry is the
    # first (and only) renewed value.
    assert _renewal_count(db_engine, key) == 1
    assert _renewal_count(db_engine) == 1
    assert _lease_row(db_engine, token)["expires_at"].isoformat() == first_body[
        "expires_at"
    ]
    assert (
        http_client.get(f"/leases/{token}").json()["expires_at"]
        == first_body["expires_at"]
    )


def test_concurrent_same_key_renews_extend_exactly_once(http_client, db_engine):
    granted = acquire(http_client, antenna_id="ANT-03", duration_seconds=60)
    token = granted.json()["lease_token"]
    original_expiry = datetime.fromisoformat(granted.json()["expires_at"])
    key = make_key("renew-storm")

    barrier = threading.Barrier(8)
    base_url = str(http_client.base_url)

    def one(_: int) -> httpx.Response:
        barrier.wait(timeout=10)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            return renew(client, token, additional_seconds=10, idempotency_key=key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(one, range(8)))

    assert all(r.status_code == 200 for r in responses), [
        r.status_code for r in responses
    ]
    bodies = [r.json() for r in responses]
    # Exactly one caller performed the renewal; the others replayed it.
    assert sum(1 for b in bodies if b["replay"] is False) == 1
    assert sum(1 for b in bodies if b["replay"] is True) == 7
    # Everyone observed the same before/after expiry.
    assert len({b["expires_at"] for b in bodies}) == 1
    assert len({b["previous_expires_at"] for b in bodies}) == 1

    assert _renewal_count(db_engine, key) == 1
    final_expiry = _lease_row(db_engine, token)["expires_at"]
    assert final_expiry == original_expiry + timedelta(seconds=10)


def test_same_key_different_seconds_is_stable_conflict(http_client, db_engine):
    token = acquire(http_client, antenna_id="ANT-04", duration_seconds=60).json()[
        "lease_token"
    ]
    key = make_key("renew")
    first = renew(http_client, token, additional_seconds=10, idempotency_key=key)
    assert first.status_code == 200
    first_body = first.json()

    for _ in range(2):
        conflict = renew(
            http_client, token, additional_seconds=11, idempotency_key=key
        )
        assert conflict.status_code == 409
        error = conflict.json()["error"]
        assert error["code"] == "IDEMPOTENCY_CONFLICT"
        assert error["details"]["idempotency_key"] == key

    # The conflict wrote nothing: the lease keeps the first renewal's expiry
    # and no second record exists.
    assert _renewal_count(db_engine) == 1
    assert _lease_row(db_engine, token)["expires_at"].isoformat() == first_body[
        "expires_at"
    ]

    # The canonical parameters still replay.
    replay = renew(http_client, token, additional_seconds=10, idempotency_key=key)
    assert replay.status_code == 200
    assert replay.json()["replay"] is True
    assert replay.json()["expires_at"] == first_body["expires_at"]


def test_same_key_under_a_different_token_is_conflict(http_client, db_engine):
    first = acquire(http_client, antenna_id="ANT-05", duration_seconds=60).json()
    second = acquire(http_client, antenna_id="ANT-06", duration_seconds=60).json()
    key = make_key("renew")

    ok = renew(
        http_client, first["lease_token"], additional_seconds=10,
        idempotency_key=key,
    )
    assert ok.status_code == 200

    conflict = renew(
        http_client, second["lease_token"], additional_seconds=10,
        idempotency_key=key,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

    # The second lease was never touched and no second record was written.
    assert _renewal_count(db_engine) == 1
    assert _lease_row(db_engine, second["lease_token"])["expires_at"].isoformat() == (
        second["expires_at"]
    )


def test_renew_unknown_token_is_404_and_writes_nothing(http_client, db_engine):
    for _ in range(2):
        resp = renew(http_client, "no-such-token", additional_seconds=10)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "LEASE_NOT_FOUND"
        assert resp.json()["error"]["details"]["lease_token"] == "no-such-token"
    assert _renewal_count(db_engine) == 0
    assert count_rows(db_engine, "SELECT count(*) FROM leases") == 0


def test_renew_expired_lease_is_rejected_and_data_is_unchanged(
    http_client, db_engine
):
    expired = insert_expired_lease(
        db_engine, antenna_id=KNOWN_ANTENNA, age_seconds=2, ttl_seconds=10
    )
    token = expired["token"]
    before = _lease_row(db_engine, token)

    for _ in range(2):
        resp = renew(http_client, token, additional_seconds=30)
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "LEASE_EXPIRED"
        assert error["details"]["lease_token"] == token
        assert error["details"]["expires_at"] == before["expires_at"].isoformat()

    # The rejection rewrote nothing: the expired row is byte-identical and no
    # renewal was recorded.
    assert _lease_row(db_engine, token) == before
    assert _renewal_count(db_engine) == 0

    # The rejection did not disturb the hand-over either: a successor takes
    # the antenna, and the old token keeps being refused afterwards.
    successor = acquire(http_client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)
    assert successor.status_code == 200
    again = renew(http_client, token, additional_seconds=30)
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "LEASE_EXPIRED"
    assert _lease_row(db_engine, token) == before
    new_token = successor.json()["lease_token"]
    assert http_client.get(f"/leases/{new_token}").json()["active"] is True
    assert _renewal_count(db_engine) == 0


def test_renew_released_lease_is_rejected_and_writes_nothing(
    http_client, db_engine
):
    token = acquire(http_client, antenna_id="ANT-02", duration_seconds=60).json()[
        "lease_token"
    ]
    released = release(http_client, token)
    assert released.status_code == 200
    released_at = released.json()["released_at"]

    resp = renew(http_client, token, additional_seconds=10)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "LEASE_EXPIRED"

    row = _lease_row(db_engine, token)
    assert row["released_at"].isoformat() == released_at
    assert _renewal_count(db_engine) == 0


def test_repeated_renewals_accumulate_from_the_current_expiry(
    http_client, db_engine
):
    granted = acquire(http_client, antenna_id="ANT-03", duration_seconds=30)
    token = granted.json()["lease_token"]
    acquired_at = datetime.fromisoformat(granted.json()["acquired_at"])
    expiry0 = datetime.fromisoformat(granted.json()["expires_at"])

    first = renew(http_client, token, additional_seconds=10)
    assert first.status_code == 200
    first_body = first.json()
    assert datetime.fromisoformat(first_body["previous_expires_at"]) == expiry0
    expiry1 = datetime.fromisoformat(first_body["expires_at"])
    assert expiry1 == expiry0 + timedelta(seconds=10)

    second = renew(http_client, token, additional_seconds=15)
    assert second.status_code == 200
    second_body = second.json()
    # The second renewal builds on the first one's result, not on "now".
    assert second_body["previous_expires_at"] == first_body["expires_at"]
    expiry2 = datetime.fromisoformat(second_body["expires_at"])
    assert expiry2 == expiry1 + timedelta(seconds=15)

    # Total extension is exactly duration + 10 + 15 measured from acquisition:
    # proof that extensions accumulate on the stored expiry.
    assert expiry2 - acquired_at == timedelta(seconds=30 + 10 + 15)
    assert _lease_row(db_engine, token)["expires_at"] == expiry2
    assert _renewal_count(db_engine) == 2
    assert http_client.get(f"/leases/{token}").json()["expires_at"] == second_body[
        "expires_at"
    ]


def test_concurrent_distinct_key_renews_serialise_and_accumulate(
    http_client, db_engine
):
    granted = acquire(http_client, antenna_id="ANT-04", duration_seconds=60)
    token = granted.json()["lease_token"]
    expiry0 = datetime.fromisoformat(granted.json()["expires_at"])

    barrier = threading.Barrier(6)
    base_url = str(http_client.base_url)

    def one(_: int) -> httpx.Response:
        barrier.wait(timeout=10)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            return renew(client, token, additional_seconds=5)

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(one, range(6)))

    assert all(r.status_code == 200 for r in responses), [
        r.status_code for r in responses
    ]
    bodies = [r.json() for r in responses]
    assert all(b["replay"] is False for b in bodies)

    # The renewals serialised on the antenna row: their before/after values
    # form one continuous chain starting at the original expiry, and the
    # final expiry is exactly original + 6 * 5s.
    pairs = {
        (b["previous_expires_at"], b["expires_at"]) for b in bodies
    }
    chain_head = expiry0.isoformat()
    for _ in range(6):
        link = next(p for p in pairs if p[0] == chain_head)
        pairs.remove(link)
        chain_head = link[1]
    assert not pairs
    assert datetime.fromisoformat(chain_head) == expiry0 + timedelta(seconds=30)
    assert _lease_row(db_engine, token)["expires_at"] == expiry0 + timedelta(
        seconds=30
    )
    assert _renewal_count(db_engine) == 6


def test_concurrent_renew_while_active_blocks_all_contenders(
    http_client, db_engine
):
    # Lease with a 1.5s fuse: a renewal racing five acquirers always wins
    # while unexpired — every contender must leave empty-handed.
    planted = insert_expired_lease(
        db_engine, antenna_id=KNOWN_ANTENNA, age_seconds=-1.5, ttl_seconds=10
    )
    token = planted["token"]
    planted_expiry = planted["expires_at"]

    kinds = ["renew"] + ["acquire"] * 5
    barrier = threading.Barrier(len(kinds))
    base_url = str(http_client.base_url)

    def run(kind: str) -> httpx.Response:
        barrier.wait(timeout=10)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            if kind == "renew":
                return renew(client, token, additional_seconds=30)
            return acquire(client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)

    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        responses = list(pool.map(run, kinds))

    renew_resp, acquire_resps = responses[0], responses[1:]
    assert renew_resp.status_code == 200, renew_resp.text
    assert renew_resp.json()["replay"] is False
    assert all(r.status_code == 409 for r in acquire_resps), [
        r.status_code for r in acquire_resps
    ]
    for r in acquire_resps:
        assert r.json()["error"]["code"] == "ANTENNA_BUSY"

    # Still the original single lease, now living 30s longer; no successor
    # was created alongside it.
    assert datetime.fromisoformat(
        renew_resp.json()["expires_at"]
    ) == planted_expiry + timedelta(seconds=30)
    assert count_rows(db_engine, "SELECT count(*) FROM leases") == 1
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1
    assert _renewal_count(db_engine) == 1


def test_expired_lease_renew_loses_the_handover_race(http_client, db_engine):
    # Lease already past its boundary: the renewal is refused and exactly one
    # of the racing acquirers takes over — never both outcomes at once.
    expired = insert_expired_lease(
        db_engine, antenna_id=KNOWN_ANTENNA, age_seconds=0.5, ttl_seconds=10
    )
    token = expired["token"]

    kinds = ["renew"] + ["acquire"] * 5
    barrier = threading.Barrier(len(kinds))
    base_url = str(http_client.base_url)

    def run(kind: str) -> httpx.Response:
        barrier.wait(timeout=10)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            if kind == "renew":
                return renew(client, token, additional_seconds=30)
            return acquire(client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)

    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        responses = list(pool.map(run, kinds))

    renew_resp, acquire_resps = responses[0], responses[1:]
    assert renew_resp.status_code == 409
    assert renew_resp.json()["error"]["code"] == "LEASE_EXPIRED"

    winners = [r for r in acquire_resps if r.status_code == 200]
    busy = [r for r in acquire_resps if r.status_code == 409]
    assert len(winners) == 1, [r.status_code for r in acquire_resps]
    assert len(winners) + len(busy) == len(acquire_resps)  # never a 5xx

    # The expired row was not revived or extended; exactly one new lease
    # holds the antenna now.
    assert _lease_row(db_engine, token)["expires_at"] == expired["expires_at"]
    assert _renewal_count(db_engine) == 0
    assert count_rows(db_engine, "SELECT count(*) FROM leases") == 2
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1


def test_boundary_race_never_produces_double_control(http_client, db_engine):
    # The lease expires 0.3s after planting; the renewal and the acquirers
    # are released from one barrier. Whichever side the database clock lands
    # on, exactly one outcome is possible: either the renewal extended the
    # lease (all acquirers busy, no new lease) or the lease was already gone
    # (renewal refused, exactly one successor).
    planted = insert_expired_lease(
        db_engine, antenna_id=KNOWN_ANTENNA, age_seconds=-0.3, ttl_seconds=10
    )
    token = planted["token"]
    planted_expiry = planted["expires_at"]

    kinds = ["renew"] + ["acquire"] * 5
    barrier = threading.Barrier(len(kinds))
    base_url = str(http_client.base_url)

    def run(kind: str) -> httpx.Response:
        barrier.wait(timeout=10)
        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            if kind == "renew":
                return renew(client, token, additional_seconds=30)
            return acquire(client, antenna_id=KNOWN_ANTENNA, duration_seconds=30)

    with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
        responses = list(pool.map(run, kinds))

    renew_resp, acquire_resps = responses[0], responses[1:]
    winners = [r for r in acquire_resps if r.status_code == 200]
    for r in acquire_resps:
        assert r.status_code in (200, 409)  # never a 5xx

    if renew_resp.status_code == 200:
        # Renewal won the boundary: no acquirer may hold the antenna.
        assert winners == []
        assert datetime.fromisoformat(
            renew_resp.json()["expires_at"]
        ) == planted_expiry + timedelta(seconds=30)
        assert count_rows(db_engine, "SELECT count(*) FROM leases") == 1
        assert _renewal_count(db_engine) == 1
    else:
        # Boundary belonged to the newcomers: renewal refused, one successor.
        assert renew_resp.status_code == 409
        assert renew_resp.json()["error"]["code"] == "LEASE_EXPIRED"
        assert len(winners) == 1
        assert _lease_row(db_engine, token)["expires_at"] == planted_expiry
        assert _renewal_count(db_engine) == 0

    # The invariant that must hold on either branch: one active lease.
    assert active_lease_count(db_engine, KNOWN_ANTENNA) == 1


def test_renew_rejection_does_not_affect_the_later_holder(http_client, db_engine):
    first = acquire(http_client, antenna_id="ANT-05", duration_seconds=5)
    token = first.json()["lease_token"]
    renewed = renew(http_client, token, additional_seconds=5)
    assert renewed.status_code == 200

    # Outlive the renewed boundary, then a successor takes over.
    deadline = datetime.fromisoformat(renewed.json()["expires_at"])
    while datetime.now(timezone.utc) <= deadline:
        time.sleep(0.1)
    time.sleep(0.3)
    successor = acquire(
        http_client, antenna_id="ANT-05", controller="next-shift",
        duration_seconds=60,
    )
    assert successor.status_code == 200
    new_token = successor.json()["lease_token"]

    # A late renewal attempt on the old token is refused and leaves the new
    # holder fully intact.
    late = renew(http_client, token, additional_seconds=30)
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "LEASE_EXPIRED"

    status = http_client.get(f"/leases/{new_token}")
    assert status.json()["active"] is True
    assert status.json()["expires_at"] == successor.json()["expires_at"]
    assert _lease_row(db_engine, new_token)["expires_at"].isoformat() == (
        successor.json()["expires_at"]
    )
    assert active_lease_count(db_engine, "ANT-05") == 1
    assert _renewal_count(db_engine) == 1  # only the original renewal


def test_renew_replay_survives_expiry_and_handover(http_client, db_engine):
    token = acquire(http_client, antenna_id="ANT-06", duration_seconds=5).json()[
        "lease_token"
    ]
    key = make_key("renew")
    first = renew(http_client, token, additional_seconds=5, idempotency_key=key)
    assert first.status_code == 200
    first_body = first.json()

    deadline = datetime.fromisoformat(first_body["expires_at"])
    while datetime.now(timezone.utc) <= deadline:
        time.sleep(0.1)
    time.sleep(0.3)
    successor = acquire(http_client, antenna_id="ANT-06", duration_seconds=30)
    assert successor.status_code == 200

    # Same key + same params after the lease is long gone: still the recorded
    # replay, never an error and never a second extension.
    replay = renew(http_client, token, additional_seconds=5, idempotency_key=key)
    assert replay.status_code == 200
    body = replay.json()
    assert body["replay"] is True
    assert body["previous_expires_at"] == first_body["previous_expires_at"]
    assert body["expires_at"] == first_body["expires_at"]
    assert _renewal_count(db_engine) == 1
    assert active_lease_count(db_engine, "ANT-06") == 1


def test_boundary_additional_seconds_accepted(http_client, db_engine):
    token = acquire(http_client, antenna_id="ANT-01", duration_seconds=60).json()[
        "lease_token"
    ]
    expiry0 = datetime.fromisoformat(
        http_client.get(f"/leases/{token}").json()["expires_at"]
    )

    low = renew(http_client, token, additional_seconds=5)
    assert low.status_code == 200, low.text
    high = renew(http_client, token, additional_seconds=120)
    assert high.status_code == 200, high.text

    final_expiry = datetime.fromisoformat(high.json()["expires_at"])
    assert final_expiry == expiry0 + timedelta(seconds=125)
    assert _renewal_count(db_engine) == 2


def test_invalid_renew_payloads_are_422_and_write_nothing(http_client, db_engine):
    token = acquire(http_client, antenna_id="ANT-02", duration_seconds=60).json()[
        "lease_token"
    ]
    before = _lease_row(db_engine, token)

    bad_payloads = [
        {"additional_seconds": 4, "idempotency_key": make_key()},
        {"additional_seconds": 121, "idempotency_key": make_key()},
        {"additional_seconds": 0, "idempotency_key": make_key()},
        {"additional_seconds": -5, "idempotency_key": make_key()},
        {"additional_seconds": 7.5, "idempotency_key": make_key()},
        {"additional_seconds": "10", "idempotency_key": make_key()},
        {"additional_seconds": None, "idempotency_key": make_key()},
        {"idempotency_key": make_key()},  # missing additional_seconds
        {"additional_seconds": 10},  # missing idempotency_key
        {"additional_seconds": 10, "idempotency_key": "   "},
        {"additional_seconds": 10, "idempotency_key": make_key(), "extra": 1},
    ]
    for payload in bad_payloads:
        resp = http_client.post(f"/leases/{token}/renew", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    # No rejection touched the lease or recorded anything.
    assert _lease_row(db_engine, token) == before
    assert _renewal_count(db_engine) == 0


def test_acquire_and_status_shapes_are_unchanged_by_the_renew_feature(
    http_client, db_engine
):
    granted = acquire(http_client, antenna_id="ANT-03", duration_seconds=30)
    assert granted.status_code == 200
    assert set(granted.json()) == ACQUIRE_RESPONSE_FIELDS
    token = granted.json()["lease_token"]

    renewed = renew(http_client, token, additional_seconds=10)
    assert renewed.status_code == 200

    status = http_client.get(f"/leases/{token}")
    assert status.status_code == 200
    assert set(status.json()) == STATUS_RESPONSE_FIELDS
    # The status query simply reflects the renewed expiry; every other
    # contract field is untouched.
    assert status.json()["expires_at"] == renewed.json()["expires_at"]
    assert status.json()["acquired_at"] == granted.json()["acquired_at"]
    assert status.json()["active"] is True
