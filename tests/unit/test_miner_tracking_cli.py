import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import miner_tracking


def test_tracking_cli_runs_from_source_checkout():
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "scripts/miner_tracking.py", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "reconcile" in result.stdout.lower()


@pytest.mark.parametrize(
    ("include_ambiguous", "expected_attach_count"),
    [(False, 0), (True, 1)],
)
def test_sync_controls_ambiguous_matches(
    monkeypatch, include_ambiguous, expected_attach_count
):
    tracker = MagicMock()
    tracker.pending_requests.return_value = [
        {
            "request_id": "request-id",
            "validator_uid": 70,
            "received_at": 100.0,
        }
    ]
    tracker.recent_requests.return_value = []
    monkeypatch.setattr(miner_tracking, "fetch_wandb_score_rows", lambda **kwargs: [])
    monkeypatch.setattr(
        miner_tracking,
        "match_score_rows",
        lambda *args, **kwargs: [
            {
                "request_id": "request-id",
                "wandb_run": "run-id",
                "task_id": "task-id",
                "adjusted_score": 0.6,
                "final_score": 0.5,
                "score_timestamp": 110.0,
                "match_confidence": "medium",
            }
        ],
    )
    args = SimpleNamespace(
        hours=10**9,
        miner_hotkey="miner-hotkey",
        miner_uid=36,
        entity="afterparty",
        project="conversationgenome",
        max_delay_seconds=900,
        include_ambiguous=include_ambiguous,
        limit=20,
    )

    miner_tracking._sync(tracker, args)

    assert tracker.attach_score.call_count == expected_attach_count
