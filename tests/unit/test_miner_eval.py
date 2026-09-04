import pytest

from scripts.miner_eval import miner_eval


def _score(validator_uid, hotkey, final, adjusted, timestamp):
    return {
        "validator_uid": validator_uid,
        "miner_uid": 73,
        "miner_hotkey": hotkey,
        "wandb_run": f"run-{validator_uid}",
        "task_id": f"task-{timestamp}",
        "adjusted_score": adjusted,
        "final_score": final,
        "score_timestamp": timestamp,
    }


def test_analyze_scores_summarizes_penalties_validators_and_hotkeys():
    rows = [
        _score(70, "old-hotkey", 0.2, 0.4, 100.0),
        _score(70, "old-hotkey", 0.4, 0.4, 200.0),
        _score(78, "replacement-hotkey", 0.6, 0.8, 300.0),
    ]

    stats = miner_eval.analyze_scores(rows)

    assert stats["total_scores"] == 3
    assert stats["mean_final_score"] == pytest.approx(0.4)
    assert stats["median_final_score"] == pytest.approx(0.4)
    assert stats["min_final_score"] == pytest.approx(0.2)
    assert stats["max_final_score"] == pytest.approx(0.6)
    assert stats["mean_adjusted_score"] == pytest.approx(1.6 / 3)
    assert stats["penalty_count"] == 2
    assert stats["penalty_percentage"] == pytest.approx(200 / 3)
    assert stats["mean_penalty"] == pytest.approx(0.2)
    assert stats["hotkey_stats"] == {
        "old-hotkey": {"count": 2, "mean_final_score": pytest.approx(0.3)},
        "replacement-hotkey": {
            "count": 1,
            "mean_final_score": pytest.approx(0.6),
        },
    }
    assert stats["validator_stats"] == {
        70: {"count": 2, "mean_final_score": pytest.approx(0.3)},
        78: {"count": 1, "mean_final_score": pytest.approx(0.6)},
    }


def test_main_uses_four_hour_window_and_reports_no_scores(monkeypatch, capsys):
    captured = {}

    def fetch_scores(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(miner_eval.time, "time", lambda: 20_000.0)
    monkeypatch.setattr(miner_eval, "fetch_wandb_score_rows", fetch_scores)

    result = miner_eval.main(["--uid", "73"])

    assert result == 0
    assert captured["miner_uid"] == 73
    assert captured["miner_hotkey"] is None
    assert captured["since_timestamp"] == 5_600.0
    assert (
        "No validator scores found for UID 73 in the last 4 hours."
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "argv",
    (
        ["--uid", "-1"],
        ["--uid", "73", "--hours", "-1"],
        ["--uid", "73", "--hours", "0"],
        ["--uid", "73", "--hours", "nan"],
        ["--uid", "73", "--hours", "inf"],
        ["--uid", "73", "--hours", "-inf"],
    ),
)
def test_main_rejects_invalid_uid_and_window(monkeypatch, argv):
    monkeypatch.setattr(miner_eval, "fetch_wandb_score_rows", lambda **kwargs: [])

    with pytest.raises(SystemExit):
        miner_eval.main(argv)


def test_main_reports_public_data_failure(monkeypatch, capsys):
    def fail(**kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(miner_eval, "fetch_wandb_score_rows", fail)

    result = miner_eval.main(["--uid", "73"])

    assert result == 1
    assert (
        "Could not fetch public W&B scores: network unavailable"
        in capsys.readouterr().err
    )
