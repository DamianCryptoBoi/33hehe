import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_miner_llm_routes
from scripts.benchmark_miner_llm_routes import assert_backend_matches


def test_openai_null_base_url_matches_official_endpoint():
    backend = type("LlmOpenAI", (), {})()
    backend.model = "gpt-5.4"
    backend.reasoning_effort = "none"
    backend.client = SimpleNamespace(base_url="https://api.openai.com/v1/")
    route = {
        "provider": "openai",
        "model": "gpt-5.4",
        "reasoning_effort": "none",
        "base_url": None,
    }

    assert_backend_matches(route, backend)


def test_backend_model_mismatch_is_rejected():
    backend = type("LlmOpenAI", (), {})()
    backend.model = "wrong-model"
    backend.reasoning_effort = "none"
    backend.client = SimpleNamespace(base_url="https://api.openai.com/v1/")
    route = {
        "provider": "openai",
        "model": "gpt-5.4",
        "reasoning_effort": "none",
    }

    with pytest.raises(RuntimeError, match="model mismatch"):
        assert_backend_matches(route, backend)


def test_load_saved_requests_samples_each_requested_task_type(tmp_path):
    db_path = tmp_path / "requests.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE miner_requests (
                request_id TEXT,
                received_at REAL,
                task_type TEXT,
                task_payload TEXT,
                status TEXT
            )
            """
        )
        rows = []
        for task_type in ("conversation_tagging", "webpage_metadata_generation"):
            rows.extend(
                (
                    f"{task_type}-{index}",
                    float(index),
                    task_type,
                    json.dumps({"type": task_type, "guid": str(index)}),
                    "success",
                )
                for index in range(6)
            )
        rows.append(
            (
                "conversation-error",
                100.0,
                "conversation_tagging",
                json.dumps({"type": "conversation_tagging", "guid": "error"}),
                "error",
            )
        )
        connection.executemany(
            "INSERT INTO miner_requests VALUES (?, ?, ?, ?, ?)", rows
        )

    load_saved_requests = getattr(
        benchmark_miner_llm_routes, "load_saved_requests", None
    )
    assert callable(load_saved_requests), "multi-task saved-request loader is missing"

    requests = load_saved_requests(
        db_path,
        {"conversation_tagging", "webpage_metadata_generation", "survey_tagging"},
        samples=5,
    )

    assert len(requests["conversation_tagging"]) == 5
    assert len(requests["webpage_metadata_generation"]) == 5
    assert requests["survey_tagging"] == []
    assert all(
        payload["type"] == task_type
        for task_type, task_requests in requests.items()
        for _, payload in task_requests
    )
    assert all(
        request_id != "conversation-error"
        for request_id, _ in requests["conversation_tagging"]
    )


def test_load_saved_requests_reuses_the_same_seeded_cohort(tmp_path):
    db_path = tmp_path / "requests.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE miner_requests (
                request_id TEXT,
                received_at REAL,
                task_type TEXT,
                task_payload TEXT,
                status TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO miner_requests VALUES (?, ?, ?, ?, ?)",
            [
                (
                    f"request-{index}",
                    float(index),
                    "conversation_tagging",
                    json.dumps({"type": "conversation_tagging", "guid": str(index)}),
                    "success",
                )
                for index in range(10)
            ],
        )

    first = benchmark_miner_llm_routes.load_saved_requests(
        db_path, {"conversation_tagging"}, samples=5, seed=42
    )
    second = benchmark_miner_llm_routes.load_saved_requests(
        db_path, {"conversation_tagging"}, samples=5, seed=42
    )

    assert first == second


def test_results_are_formatted_as_a_table():
    format_results_table = getattr(
        benchmark_miner_llm_routes, "format_results_table", None
    )
    assert callable(format_results_table), "results table formatter is missing"

    table = format_results_table(
        [
            {
                "task_type": "conversation_tagging",
                "provider": "openai",
                "model": "gpt-5.2",
                "reasoning_effort": None,
                "deadline": 12.0,
                "completed": 2,
                "target": 2,
                "durations": [1.0, 2.0],
                "attempt_durations": [1.0, 2.0],
                "status": "OK",
            },
            {
                "task_type": "survey_tagging",
                "provider": "openai",
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
                "deadline": None,
                "completed": 0,
                "target": 5,
                "durations": [],
                "attempt_durations": [],
                "status": "MISSING",
            },
        ]
    )

    assert (
        "| conversation_tagging | openai/gpt-5.2 | provider default | 12.000 | "
        "2/2 | 1.000 | 1.500 | 1.950 | 2.000 | 10.050 | 0 | OK |"
        in table
    )
    assert (
        "| survey_tagging | openai/gpt-5.4 | medium | - | 0/5 | - | - | - | - | - | - | MISSING |"
        in table
    )


def test_empty_miner_result_is_rejected():
    is_valid_miner_result = getattr(
        benchmark_miner_llm_routes, "is_valid_miner_result", None
    )
    assert callable(is_valid_miner_result), "miner-result validator is missing"

    assert not is_valid_miner_result({"tags": [], "vectors": None})
    assert is_valid_miner_result({"tags": ["example"], "vectors": None})
    assert not is_valid_miner_result(None)


@pytest.mark.asyncio
async def test_benchmark_reports_deadline_headroom_and_misses(
    monkeypatch, capsys
):
    from conversationgenome.llm import llm_factory
    from conversationgenome.miner.MinerLib import MinerLib
    from conversationgenome.task import task_factory

    task_type = "conversation_tagging"
    route = {"provider": "openai", "model": "gpt-5.2"}
    config = {"default": route, "tasks": {}}

    backend = type("LlmOpenAI", (), {})()
    backend.model = "gpt-5.2"
    backend.reasoning_effort = None
    backend.client = SimpleNamespace(base_url="https://api.openai.com/v1/")

    monkeypatch.setattr(
        benchmark_miner_llm_routes,
        "load_saved_requests",
        lambda _db, task_types, _samples, _seed=0: {
            candidate: [
                ("request-1", {"type": candidate, "timeout": 12.0}),
                ("request-2", {"type": candidate, "timeout": 24.0}),
                ("request-3", {"type": candidate, "timeout": 24.0}),
            ]
            if candidate == task_type
            else []
            for candidate in task_types
        },
    )
    monkeypatch.setattr(llm_factory, "load_miner_llm_config", lambda: config)
    monkeypatch.setattr(llm_factory, "get_llm_backend", lambda **_kwargs: backend)
    monkeypatch.setattr(
        task_factory,
        "parse_task",
        lambda payload: SimpleNamespace(
            type=payload["type"], timeout=payload["timeout"]
        ),
    )

    async def mine(_self, task):
        if task.timeout == 24.0 and not getattr(mine, "failed_once", False):
            mine.failed_once = True
            raise RuntimeError("simulated model failure")
        return {"tags": ["example"], "vectors": None}

    monkeypatch.setattr(MinerLib, "do_mining", mine)
    clock = iter([100.0, 115.0, 200.0, 215.0, 300.0, 305.0])
    monkeypatch.setattr(
        benchmark_miner_llm_routes.time, "perf_counter", lambda: next(clock)
    )

    await benchmark_miner_llm_routes.benchmark(
        3, Path("saved.sqlite3"), {task_type}
    )

    output = capsys.readouterr().out
    assert (
        "| conversation_tagging | openai/gpt-5.2 | provider default | "
        "12.000-24.000 | 2/3 | 5.000 | 11.667 | 15.000 | 15.000 | "
        "-1.800 | 1 | FAILED 1; MISSED 1 |"
        in output
    )
    assert "webpage_metadata_generation" not in output


def test_invalid_saved_task_does_not_print_conversation_content(tmp_path):
    db_path = tmp_path / "requests.sqlite3"
    sentinel = "PRIVATE_CONVERSATION_SENTINEL"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE miner_requests (
                request_id TEXT,
                received_at REAL,
                task_type TEXT,
                task_payload TEXT,
                status TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO miner_requests VALUES (?, ?, ?, ?, ?)",
            (
                "invalid-request",
                100.0,
                "conversation_tagging",
                json.dumps(
                    {
                        "type": "conversation_tagging",
                        "mode": "mine",
                        "input": sentinel,
                    }
                ),
                "success",
            ),
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_miner_llm_routes",
            "--db",
            str(db_path),
            "--task",
            "conversation_tagging",
        ],
        cwd=benchmark_miner_llm_routes.REPOSITORY,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENAI_API_KEY": "test-key"},
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0
    assert "FAILED (InvalidSavedTask)" in output
    assert "| conversation_tagging |" in output
    assert "| PARTIAL; FAILED 1 |" in output
    assert sentinel not in output
