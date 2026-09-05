#!/usr/bin/env python3
"""Benchmark miner task types with reproducibly sampled saved requests."""

import argparse
import asyncio
import json
import os
import random
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
    db_path: Path, task_types: set[str], samples: int, seed: int = 0
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
                ORDER BY request_id
                """,
                (task_type,),
            ).fetchall()
            random.Random(f"{seed}:{task_type}").shuffle(rows)
            rows = rows[:samples]
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
    return isinstance(result, dict) and any(result.values())


def _percentile(values: list[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


def format_results_table(results: list[dict]) -> str:
    lines = [
        "| task_type | provider/model | thinking | deadline_s | completed | min_s | avg_s | p95_s | max_s | p05_headroom_s | misses | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        attempts = result.get("attempts")
        if attempts is None:
            deadline = result.get("deadline")
            attempts = [
                (duration, deadline)
                for duration in result.get("attempt_durations", result["durations"])
            ]
        attempt_durations = [duration for duration, _ in attempts]
        deadlines = sorted({deadline for _, deadline in attempts if deadline is not None})
        if attempt_durations:
            minimum = f"{min(attempt_durations):.3f}"
            average = f"{statistics.mean(attempt_durations):.3f}"
            p95_value = _percentile(attempt_durations, 95)
            p95 = f"{p95_value:.3f}"
            maximum = f"{max(attempt_durations):.3f}"
        else:
            minimum = average = p95 = maximum = "-"
        if len(deadlines) == 1:
            deadline_text = f"{deadlines[0]:.3f}"
        elif deadlines:
            deadline_text = f"{deadlines[0]:.3f}-{deadlines[-1]:.3f}"
        else:
            deadline_text = "-"
        headrooms = [deadline - duration for duration, deadline in attempts if deadline is not None]
        headroom = f"{_percentile(headrooms, 5):.3f}" if headrooms else "-"
        misses = str(sum(duration > deadline for duration, deadline in attempts if deadline is not None)) if deadlines else "-"
        lines.append(
            f"| {result['task_type']} | {result['provider']}/{result['model']} | "
            f"{result.get('reasoning_effort') or 'provider default'} | {deadline_text} | "
            f"{result['completed']}/{result['target']} | {minimum} | {average} | "
            f"{p95} | {maximum} | {headroom} | {misses} | {result['status']} |"
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


async def benchmark(
    samples: int,
    db_path: Path,
    task_types: set[str] | None = None,
    seed: int = 0,
) -> None:
    # Keep miner imports here so argparse can handle --help before Bittensor sees it.
    from conversationgenome.llm.llm_factory import MINER_TASK_TYPES
    from conversationgenome.llm.llm_factory import get_llm_backend
    from conversationgenome.llm.llm_factory import load_miner_llm_config
    from conversationgenome.miner.MinerLib import MinerLib
    from conversationgenome.task.task_factory import parse_task

    task_types = set(task_types or MINER_TASK_TYPES)
    unknown_task_types = task_types - MINER_TASK_TYPES
    if unknown_task_types:
        raise ValueError(f"Unknown task type(s): {', '.join(sorted(unknown_task_types))}")

    config = load_miner_llm_config()
    saved_requests = load_saved_requests(db_path, task_types, samples, seed)
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
        print(
            f"Using seeded saved requests from {Path(db_path).expanduser()} "
            f"(seed={seed})"
        )
        print("Resolved real-miner routing:")
        for task_type in sorted(task_types):
            route = config["tasks"].get(task_type, config["default"])
            backend = get_llm_backend(task_type=task_type)
            assert_backend_matches(route, backend)
            print(
                f"  {task_type:31} -> {route['provider']}/{route['model']} "
                f"thinking={route.get('reasoning_effort', 'provider default')}"
            )

        for task_type in sorted(task_types):
            route = config["tasks"].get(task_type, config["default"])
            requests = parsed_requests[task_type]
            durations = []
            attempts = []
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
                duration = None
                started_at = time.perf_counter()
                try:
                    result = await MinerLib().do_mining(task)
                    duration = time.perf_counter() - started_at
                    attempts.append((duration, task.timeout))
                    if not is_valid_miner_result(result):
                        raise RuntimeError(
                            "MinerLib.do_mining() returned an empty or invalid result"
                        )
                    durations.append(duration)
                    print(
                        f"  run {run_number}/{len(requests)} {request_id}: "
                        f"{duration:.3f}s"
                    )
                except Exception as error:
                    if duration is None:
                        attempts.append((time.perf_counter() - started_at, task.timeout))
                    failures += 1
                    print(
                        f"  run {run_number}/{len(requests)} {request_id}: "
                        f"FAILED ({type(error).__name__})"
                    )

            deadline_misses = sum(
                duration > deadline
                for duration, deadline in attempts
                if deadline is not None
            )
            if not requests:
                status = "MISSING"
            else:
                status_parts = []
                if len(requests) < samples:
                    status_parts.append("PARTIAL")
                if failures:
                    status_parts.append(f"FAILED {failures}")
                if deadline_misses:
                    status_parts.append(f"MISSED {deadline_misses}")
                status = "; ".join(status_parts) or "OK"
            results.append(
                {
                    "task_type": task_type,
                    "provider": route["provider"],
                    "model": route["model"],
                    "reasoning_effort": route.get("reasoning_effort"),
                    "completed": len(durations),
                    "target": samples,
                    "durations": durations,
                    "attempts": attempts,
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
        help="Saved requests per task type (default: 5)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_REQUEST_DB,
        help=f"Saved miner request database (default: {DEFAULT_REQUEST_DB})",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Benchmark only this task type; repeat for multiple tasks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to select a reproducible request cohort (default: 0)",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    asyncio.run(
        benchmark(
            args.samples,
            args.db,
            set(args.task) if args.task else None,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
