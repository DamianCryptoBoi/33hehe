#!/usr/bin/env python3
"""Benchmark every miner task type with random saved requests."""

import argparse
import asyncio
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST_DB = REPOSITORY / "uid217-miner-requests-latest.sqlite3"
BACKEND_CLASSES = {
    "anthropic": "LlmAnthropic",
    "chutes": "LlmChutes",
    "groq": "LlmGroq",
    "openai": "LlmOpenAI",
    "openrouter": "LlmOpenRouter",
    "vertex": "LlmVertex",
}


def load_saved_requests(
    db_path: Path, task_types: set[str], samples: int
) -> dict[str, list[tuple[str, dict | None]]]:
    db_path = Path(db_path).expanduser()
    if not db_path.is_file():
        raise FileNotFoundError(f"Saved miner request database not found: {db_path}")

    connection = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        output = {}
        for task_type in sorted(task_types):
            rows = connection.execute(
                """
                SELECT request_id, task_payload
                FROM miner_requests
                WHERE task_type = ? AND status = 'success'
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (task_type, samples),
            ).fetchall()
            output[task_type] = []
            for request_id, payload_text in rows:
                try:
                    payload = json.loads(payload_text)
                except (TypeError, json.JSONDecodeError):
                    payload = None
                if not isinstance(payload, dict) or payload.get("type") != task_type:
                    payload = None
                output[task_type].append((request_id, payload))
    except sqlite3.DatabaseError as error:
        raise RuntimeError(
            f"Could not read saved miner requests from {db_path}: {error}"
        ) from None
    finally:
        connection.close()
    return output


def is_valid_miner_result(result) -> bool:
    return isinstance(result, dict)


def format_results_table(results: list[dict]) -> str:
    lines = [
        "| task_type | provider/model | completed | min_s | avg_s | max_s | status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        durations = result["durations"]
        if durations:
            minimum = f"{min(durations):.3f}"
            average = f"{statistics.mean(durations):.3f}"
            maximum = f"{max(durations):.3f}"
        else:
            minimum = average = maximum = "-"
        lines.append(
            f"| {result['task_type']} | {result['provider']}/{result['model']} | "
            f"{result['completed']}/{result['target']} | {minimum} | {average} | "
            f"{maximum} | {result['status']} |"
        )
    return "\n".join(lines)


def assert_backend_matches(route: dict, backend) -> None:
    provider = route["provider"]
    actual_backend = backend.__class__.__name__
    if actual_backend != BACKEND_CLASSES[provider]:
        raise RuntimeError(
            f"Provider mismatch: expected {provider}, got {actual_backend}"
        )
    if backend.model != route["model"]:
        raise RuntimeError(
            f"model mismatch: expected {route['model']}, got {backend.model}"
        )

    expected_effort = route.get("reasoning_effort")
    if provider == "vertex" and expected_effort:
        expected_effort = expected_effort.upper()
    actual_effort = getattr(backend, "reasoning_effort", None)
    if actual_effort != expected_effort:
        raise RuntimeError(
            f"Reasoning mismatch: expected {expected_effort}, got {actual_effort}"
        )

    if provider == "openai":
        actual_base_url = str(backend.client.base_url).rstrip("/")
        expected_base_url = (
            route.get("base_url") or "https://api.openai.com/v1"
        ).rstrip("/")
        if actual_base_url != expected_base_url:
            raise RuntimeError(
                f"OpenAI base URL mismatch: expected {expected_base_url}, "
                f"got {actual_base_url}"
            )


async def benchmark(samples: int, db_path: Path) -> None:
    # Keep miner imports here so argparse can handle --help before Bittensor sees it.
    from conversationgenome.llm.llm_factory import MINER_TASK_TYPES
    from conversationgenome.llm.llm_factory import get_llm_backend
    from conversationgenome.llm.llm_factory import load_miner_llm_config
    from conversationgenome.miner.MinerLib import MinerLib
    from conversationgenome.task.task_factory import parse_task

    config = load_miner_llm_config()
    saved_requests = load_saved_requests(db_path, MINER_TASK_TYPES, samples)
    parsed_requests = {}
    for task_type, requests in saved_requests.items():
        parsed_requests[task_type] = []
        for request_id, task_payload in requests:
            try:
                task = parse_task(task_payload)
            except Exception:
                task = None
            parsed_requests[task_type].append((request_id, task))

    overridden_env = (
        "LLM_TYPE_OVERRIDE",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    )
    original_env = {name: os.environ.get(name) for name in overridden_env}

    # Deliberately conflicting legacy overrides prove that routed JSON wins.
    os.environ["LLM_TYPE_OVERRIDE"] = "vertex"
    os.environ["OPENAI_MODEL"] = "env-must-not-win"
    os.environ["OPENAI_BASE_URL"] = "https://invalid.example/v1"

    results = []
    try:
        print(f"Using random saved requests from {Path(db_path).expanduser()}")
        print("Resolved real-miner routing:")
        for task_type in sorted(MINER_TASK_TYPES):
            route = config["tasks"].get(task_type, config["default"])
            backend = get_llm_backend(task_type=task_type)
            assert_backend_matches(route, backend)
            print(
                f"  {task_type:31} -> {route['provider']}/{route['model']} "
                f"thinking={route.get('reasoning_effort', 'provider default')}"
            )

        for task_type in sorted(MINER_TASK_TYPES):
            route = config["tasks"].get(task_type, config["default"])
            requests = parsed_requests[task_type]
            durations = []
            failures = 0
            print(
                f"\nBenchmarking {task_type}: {route['provider']}/{route['model']} "
                f"({len(requests)}/{samples} saved requests)"
            )

            for run_number, (request_id, task) in enumerate(requests, start=1):
                if task is None:
                    failures += 1
                    print(
                        f"  run {run_number}/{len(requests)} {request_id}: "
                        "FAILED (InvalidSavedTask)"
                    )
                    continue
                try:
                    started_at = time.perf_counter()
                    result = await MinerLib().do_mining(task)
                    duration = time.perf_counter() - started_at
                    if not is_valid_miner_result(result):
                        raise RuntimeError(
                            "MinerLib.do_mining() returned a non-dictionary result"
                        )
                    durations.append(duration)
                    print(
                        f"  run {run_number}/{len(requests)} {request_id}: "
                        f"{duration:.3f}s"
                    )
                except Exception as error:
                    failures += 1
                    print(
                        f"  run {run_number}/{len(requests)} {request_id}: "
                        f"FAILED ({type(error).__name__})"
                    )

            if not requests:
                status = "MISSING"
            else:
                status_parts = []
                if len(requests) < samples:
                    status_parts.append("PARTIAL")
                if failures:
                    status_parts.append(f"FAILED {failures}")
                status = "; ".join(status_parts) or "OK"
            results.append(
                {
                    "task_type": task_type,
                    "provider": route["provider"],
                    "model": route["model"],
                    "completed": len(durations),
                    "target": samples,
                    "durations": durations,
                    "status": status,
                }
            )
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("\nSpeed summary:")
    print(format_results_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark every miner task type through MinerLib.do_mining()."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Random saved requests per task type (default: 5)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_REQUEST_DB,
        help=f"Saved miner request database (default: {DEFAULT_REQUEST_DB})",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    asyncio.run(benchmark(args.samples, args.db))


if __name__ == "__main__":
    main()
