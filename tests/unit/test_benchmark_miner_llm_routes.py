import json
import os
import sqlite3
import subprocess
import sys
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
                "completed": 2,
                "target": 2,
                "durations": [1.0, 2.0],
                "status": "OK",
            },
            {
                "task_type": "survey_tagging",
                "provider": "openai",
                "model": "gpt-5.4",
                "completed": 0,
                "target": 5,
                "durations": [],
                "status": "MISSING",
            },
        ]
    )

    assert (
        "| conversation_tagging | openai/gpt-5.2 | 2/2 | 1.000 | 1.500 | 2.000 | OK |"
        in table
    )
    assert "| survey_tagging | openai/gpt-5.4 | 0/5 | - | - | - | MISSING |" in table


def test_empty_dictionary_is_a_completed_miner_result():
    is_valid_miner_result = getattr(
        benchmark_miner_llm_routes, "is_valid_miner_result", None
    )
    assert callable(is_valid_miner_result), "miner-result validator is missing"

    assert is_valid_miner_result({"tags": [], "vectors": None})
    assert not is_valid_miner_result(None)


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
        ],
        cwd=benchmark_miner_llm_routes.REPOSITORY,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENAI_API_KEY": "test-key"},
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0
    assert "FAILED (InvalidSavedTask)" in output
    assert (
        "| conversation_tagging | openai/gpt-5.2 | 0/5 | - | - | - | "
        "PARTIAL; FAILED 1 |"
    ) in output
    assert sentinel not in output
