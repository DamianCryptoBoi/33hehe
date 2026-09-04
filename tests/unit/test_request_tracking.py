import json
import stat


def _tracking_module():
    from conversationgenome.miner import tracking

    return tracking


def test_tracker_records_redacted_request_and_answer(tmp_path):
    tracking = _tracking_module()
    tracker = tracking.RequestTracker(tmp_path / "requests.sqlite3")

    request_id = tracker.start_request(
        validator_hotkey="validator-hotkey",
        validator_uid=70,
        miner_hotkey="miner-hotkey",
        miner_uid=36,
        task_type="conversation_tagging",
        task_timeout=12,
        task_payload={
            "input": {"lines": [[0, "hello"]]},
            "authorization": "Bearer secret",
        },
        received_at=100.0,
    )
    tracker.finish_request(
        request_id,
        answer={"tags": ["greeting", "conversation"]},
        status="success",
        completed_at=102.5,
    )

    record = tracker.get_request(request_id)
    payload = json.loads(record["task_payload"])
    answer = json.loads(record["answer"])

    assert payload["authorization"] == "[REDACTED]"
    assert payload["input"]["lines"] == [[0, "hello"]]
    assert answer == {"tags": ["greeting", "conversation"]}
    assert record["duration_ms"] == 2500
    assert record["status"] == "success"
    assert record["input_hash"]
    assert stat.S_IMODE(tracker.db_path.stat().st_mode) == 0o600


def test_tracker_redacts_token_shaped_strings_under_unknown_keys(tmp_path):
    tracking = _tracking_module()
    tracker = tracking.RequestTracker(tmp_path / "requests.sqlite3")

    request_id = tracker.start_request(
        validator_hotkey="validator-hotkey",
        validator_uid=70,
        miner_hotkey="miner-hotkey",
        miner_uid=36,
        task_type="conversation_tagging",
        task_timeout=12,
        task_payload={
            "note": "Authorization: Bearer validator-secret",
            "debug": "upstream key sk-example-secret",
        },
        received_at=100.0,
    )

    payload = json.loads(tracker.get_request(request_id)["task_payload"])

    assert "validator-secret" not in payload["note"]
    assert "sk-example-secret" not in payload["debug"]


def test_score_matching_uses_validator_and_closest_preceding_request():
    tracking = _tracking_module()
    requests = [
        {"request_id": "v70-old", "validator_uid": 70, "received_at": 100.0},
        {"request_id": "v70-new", "validator_uid": 70, "received_at": 140.0},
        {"request_id": "v78", "validator_uid": 78, "received_at": 145.0},
    ]
    scores = [
        {
            "validator_uid": 70,
            "score_timestamp": 150.0,
            "task_id": "task-70",
            "final_score": 0.61,
            "adjusted_score": 0.61,
        },
        {
            "validator_uid": 78,
            "score_timestamp": 160.0,
            "task_id": "task-78",
            "final_score": 0.58,
            "adjusted_score": 0.60,
        },
    ]

    matches = tracking.match_score_rows(requests, scores, max_delay_seconds=120)

    assert [match["request_id"] for match in matches] == ["v70-new", "v78"]
    assert matches[0]["match_confidence"] == "medium"
    assert matches[1]["match_confidence"] == "high"


def test_tracker_attaches_reconciled_score(tmp_path):
    tracking = _tracking_module()
    tracker = tracking.RequestTracker(tmp_path / "requests.sqlite3")
    request_id = tracker.start_request(
        validator_hotkey="validator-hotkey",
        validator_uid=70,
        miner_hotkey="miner-hotkey",
        miner_uid=36,
        task_type="conversation_tagging",
        task_timeout=12,
        task_payload={"type": "conversation_tagging"},
        received_at=100.0,
    )

    tracker.attach_score(
        request_id,
        wandb_run="run-id",
        task_id="task-id",
        adjusted_score=0.64,
        final_score=0.61,
        score_timestamp=150.0,
        match_confidence="high",
        matched_at=151.0,
    )

    record = tracker.get_request(request_id)
    assert record["wandb_run"] == "run-id"
    assert record["wandb_task_id"] == "task-id"
    assert record["adjusted_score"] == 0.64
    assert record["final_score"] == 0.61
    assert record["match_confidence"] == "high"


def test_tracker_lists_latest_requests_first(tmp_path):
    tracking = _tracking_module()
    tracker = tracking.RequestTracker(tmp_path / "requests.sqlite3")
    for received_at in (100.0, 200.0):
        tracker.start_request(
            validator_hotkey="validator-hotkey",
            validator_uid=70,
            miner_hotkey="miner-hotkey",
            miner_uid=36,
            task_type="conversation_tagging",
            task_timeout=12,
            task_payload={"received_at": received_at},
            received_at=received_at,
        )

    records = tracker.recent_requests(limit=1)

    assert len(records) == 1
    assert records[0]["received_at"] == 200.0


def test_fetch_wandb_scores_filters_recycled_uid(monkeypatch):
    tracking = _tracking_module()
    responses = iter(
        [
            {
                "project": {
                    "runs": {
                        "edges": [
                            {
                                "node": {
                                    "name": "run-70",
                                    "displayName": "cgp/validator-70-2.38.75-123",
                                }
                            },
                            {
                                "node": {
                                    "name": "run-78",
                                    "displayName": "cgp/validator-78-2.38.75-123",
                                }
                            },
                        ]
                    }
                }
            },
            {
                "project": {
                    "run": {
                        "sampledHistory": [
                            [
                                {
                                    "_timestamp": 150.0,
                                    "task_id.36": "ours",
                                    "hotkey.36": "miner-hotkey",
                                    "adjusted_score.36": 0.64,
                                    "final_miner_score.36": 0.61,
                                },
                                {
                                    "_timestamp": 160.0,
                                    "task_id.36": "replacement",
                                    "hotkey.36": "other-hotkey",
                                    "adjusted_score.36": 0.7,
                                    "final_miner_score.36": 0.7,
                                },
                            ]
                        ]
                    }
                }
            },
        ]
    )
    monkeypatch.setattr(tracking, "_graphql", lambda *args, **kwargs: next(responses))

    rows = tracking.fetch_wandb_score_rows(
        miner_hotkey="miner-hotkey",
        miner_uid=36,
        validator_uids={70},
        since_timestamp=100.0,
    )

    assert rows == [
        {
            "validator_uid": 70,
            "miner_uid": 36,
            "miner_hotkey": "miner-hotkey",
            "wandb_run": "run-70",
            "task_id": "ours",
            "adjusted_score": 0.64,
            "final_score": 0.61,
            "score_timestamp": 150.0,
        }
    ]


def test_fetch_wandb_scores_can_analyze_uid_without_hotkey(monkeypatch):
    tracking = _tracking_module()
    responses = iter(
        [
            {
                "project": {
                    "runs": {
                        "edges": [
                            {
                                "node": {
                                    "name": "run-70",
                                    "displayName": "cgp/validator-70-2.38.75-123000",
                                }
                            }
                        ]
                    }
                }
            },
            {
                "project": {
                    "run": {
                        "sampledHistory": [
                            [
                                {
                                    "_timestamp": 150.0,
                                    "task_id.73": "task-1",
                                    "hotkey.73": "historical-hotkey",
                                    "adjusted_score.73": 0.5,
                                    "final_miner_score.73": 0.4,
                                }
                            ]
                        ]
                    }
                }
            },
        ]
    )
    monkeypatch.setattr(tracking, "_graphql", lambda *args, **kwargs: next(responses))

    rows = tracking.fetch_wandb_score_rows(
        miner_uid=73,
        miner_hotkey=None,
        validator_uids=None,
        since_timestamp=100.0,
    )

    assert rows == [
        {
            "validator_uid": 70,
            "miner_uid": 73,
            "miner_hotkey": "historical-hotkey",
            "wandb_run": "run-70",
            "task_id": "task-1",
            "adjusted_score": 0.5,
            "final_score": 0.4,
            "score_timestamp": 150.0,
        }
    ]


def test_wandb_run_selection_keeps_window_and_one_boundary_run():
    tracking = _tracking_module()
    run_nodes = [
        {
            "node": {
                "name": "current-70",
                "displayName": "cgp/validator-70-2.38.75-1700000200000",
            }
        },
        {
            "node": {
                "name": "boundary-70",
                "displayName": "cgp/validator-70-2.38.75-1699999900000",
            }
        },
        {
            "node": {
                "name": "stale-70",
                "displayName": "cgp/validator-70-2.38.75-1699900000000",
            }
        },
        {
            "node": {
                "name": "current-78",
                "displayName": "cgp/validator-78-2.38.75-1700000300000",
            }
        },
        {
            "node": {
                "name": "other-validator",
                "displayName": "cgp/validator-12-2.38.75-1700000300000",
            }
        },
    ]

    selected = tracking._select_validator_runs(
        run_nodes,
        validator_uids={70, 78},
        since_timestamp=1_700_000_000,
    )

    assert selected == [
        (70, "current-70"),
        (70, "boundary-70"),
        (78, "current-78"),
    ]


def test_wandb_run_selection_filters_other_subnets():
    tracking = _tracking_module()
    run_nodes = [
        {
            "node": {
                "name": "mainnet",
                "displayName": "cgp/validator-70-2.38.75-1700000200000",
                "config": '{"netuid":{"value":33}}',
            }
        },
        {
            "node": {
                "name": "testnet",
                "displayName": "cgp/validator-70-2.38.75-1700000300000",
                "config": '{"netuid":{"value":138}}',
            }
        },
    ]

    selected = tracking._select_validator_runs(
        run_nodes,
        validator_uids=None,
        since_timestamp=1_700_000_000,
        netuid=33,
    )

    assert selected == [(70, "mainnet")]
