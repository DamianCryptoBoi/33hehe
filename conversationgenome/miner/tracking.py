import hashlib
import json
import re
import sqlite3
import time
import urllib.request
import uuid
from pathlib import Path


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "token",
}
_SENSITIVE_TEXT = (
    re.compile(r"(?i)(authorization\s*:\s*)?bearer\s+[^\s\"']+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
)


def _redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_KEYS or normalized.endswith(
                ("_api_key", "_token", "_secret", "_password")
            ):
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump(mode="json"))
    if isinstance(value, str):
        for pattern in _SENSITIVE_TEXT:
            value = pattern.sub("[REDACTED]", value)
    return value


def _json_text(value):
    return json.dumps(
        _redact(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


class RequestTracker:
    def __init__(self, db_path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS miner_requests (
                    request_id TEXT PRIMARY KEY,
                    received_at REAL NOT NULL,
                    completed_at REAL,
                    validator_hotkey TEXT,
                    validator_uid INTEGER,
                    miner_hotkey TEXT,
                    miner_uid INTEGER,
                    task_type TEXT,
                    task_timeout REAL,
                    task_payload TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    answer TEXT,
                    duration_ms INTEGER,
                    status TEXT NOT NULL,
                    error TEXT,
                    wandb_run TEXT,
                    wandb_task_id TEXT,
                    adjusted_score REAL,
                    final_score REAL,
                    score_timestamp REAL,
                    match_confidence TEXT,
                    matched_at REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_miner_requests_pending "
                "ON miner_requests(miner_hotkey, miner_uid, validator_uid, received_at)"
            )
        self.db_path.chmod(0o600)

    def start_request(
        self,
        *,
        validator_hotkey,
        validator_uid,
        miner_hotkey,
        miner_uid,
        task_type,
        task_timeout,
        task_payload,
        received_at=None,
    ):
        request_id = uuid.uuid4().hex
        received_at = time.time() if received_at is None else float(received_at)
        payload_text = _json_text(task_payload)
        input_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO miner_requests (
                    request_id, received_at, validator_hotkey, validator_uid,
                    miner_hotkey, miner_uid, task_type, task_timeout,
                    task_payload, input_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    request_id,
                    received_at,
                    validator_hotkey,
                    validator_uid,
                    miner_hotkey,
                    miner_uid,
                    task_type,
                    task_timeout,
                    payload_text,
                    input_hash,
                ),
            )
        return request_id

    def finish_request(
        self,
        request_id,
        *,
        answer=None,
        status="success",
        error=None,
        completed_at=None,
    ):
        completed_at = time.time() if completed_at is None else float(completed_at)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT received_at FROM miner_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown request_id: {request_id}")
            duration_ms = round(max(0, completed_at - row["received_at"]) * 1000)
            connection.execute(
                """
                UPDATE miner_requests
                SET completed_at = ?, answer = ?, duration_ms = ?, status = ?, error = ?
                WHERE request_id = ?
                """,
                (
                    completed_at,
                    _json_text(answer) if answer is not None else None,
                    duration_ms,
                    status,
                    _redact(str(error)) if error else None,
                    request_id,
                ),
            )

    def attach_score(
        self,
        request_id,
        *,
        wandb_run,
        task_id,
        adjusted_score,
        final_score,
        score_timestamp,
        match_confidence,
        matched_at=None,
    ):
        matched_at = time.time() if matched_at is None else float(matched_at)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE miner_requests
                SET wandb_run = ?, wandb_task_id = ?, adjusted_score = ?,
                    final_score = ?, score_timestamp = ?, match_confidence = ?,
                    matched_at = ?
                WHERE request_id = ?
                """,
                (
                    wandb_run,
                    task_id,
                    adjusted_score,
                    final_score,
                    score_timestamp,
                    match_confidence,
                    matched_at,
                    request_id,
                ),
            )

    def get_request(self, request_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM miner_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return dict(row) if row else None

    def pending_requests(self, *, miner_hotkey, miner_uid, since_timestamp=0):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM miner_requests
                WHERE miner_hotkey = ? AND miner_uid = ? AND received_at >= ?
                  AND score_timestamp IS NULL
                ORDER BY received_at
                """,
                (miner_hotkey, miner_uid, since_timestamp),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_requests(self, limit=20):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM miner_requests ORDER BY received_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]


def match_score_rows(request_rows, score_rows, max_delay_seconds=900):
    unmatched = {row["request_id"]: row for row in request_rows}
    matches = []
    for score in sorted(score_rows, key=lambda row: row["score_timestamp"]):
        candidates = [
            request
            for request in unmatched.values()
            if request.get("validator_uid") == score.get("validator_uid")
            and request["received_at"] <= score["score_timestamp"]
            and score["score_timestamp"] - request["received_at"] <= max_delay_seconds
        ]
        if not candidates:
            continue
        request = max(candidates, key=lambda row: row["received_at"])
        match = dict(score)
        match["request_id"] = request["request_id"]
        match["match_confidence"] = "high" if len(candidates) == 1 else "medium"
        matches.append(match)
        unmatched.pop(request["request_id"])
    return matches


def _graphql(endpoint, query, variables):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "cgp-miner-score-reconciler",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload["data"]


def _select_validator_runs(run_nodes, validator_uids, since_timestamp):
    candidates = []
    for index, edge in enumerate(run_nodes):
        run = edge["node"]
        display_name = run.get("displayName") or ""
        validator_match = re.search(r"validator-(\d+)-", display_name)
        if not validator_match:
            continue
        validator_uid = int(validator_match.group(1))
        if validator_uid not in validator_uids:
            continue
        timestamp_match = re.search(r"-(\d{13})$", display_name)
        started_at = int(timestamp_match.group(1)) / 1000 if timestamp_match else None
        candidates.append((index, validator_uid, run["name"], started_at))

    boundary_runs = {}
    for candidate in candidates:
        _, validator_uid, _, started_at = candidate
        if started_at is None or started_at >= since_timestamp:
            continue
        previous = boundary_runs.get(validator_uid)
        if previous is None or started_at > previous[3]:
            boundary_runs[validator_uid] = candidate

    selected = [
        candidate
        for candidate in candidates
        if candidate[3] is None
        or candidate[3] >= since_timestamp
        or boundary_runs.get(candidate[1]) == candidate
    ]
    return [(uid, run_name) for _, uid, run_name, _ in selected]


def fetch_wandb_score_rows(
    *,
    miner_hotkey,
    miner_uid,
    validator_uids,
    since_timestamp,
    entity="afterparty",
    project="conversationgenome",
    endpoint="https://api.wandb.ai/graphql",
):
    runs_query = """
    query Runs($project: String!, $entity: String!, $first: Int!) {
      project(name: $project, entityName: $entity) {
        runs(first: $first) { edges { node { name displayName } } }
      }
    }
    """
    run_nodes = _graphql(
        endpoint,
        runs_query,
        {"project": project, "entity": entity, "first": 500},
    )["project"]["runs"]["edges"]

    selected_runs = _select_validator_runs(
        run_nodes,
        validator_uids=validator_uids,
        since_timestamp=since_timestamp,
    )

    history_query = """
    query RunSampledHistory(
      $project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!
    ) {
      project(name: $project, entityName: $entity) {
        run(name: $name) { sampledHistory(specs: $specs) }
      }
    }
    """
    uid = int(miner_uid)
    spec = json.dumps(
        {
            "keys": [
                "_timestamp",
                f"task_id.{uid}",
                f"hotkey.{uid}",
                f"adjusted_score.{uid}",
                f"final_miner_score.{uid}",
            ],
            "samples": 50000,
        }
    )
    scores = {}
    for validator_uid, run_name in selected_runs:
        data = _graphql(
            endpoint,
            history_query,
            {
                "project": project,
                "entity": entity,
                "name": run_name,
                "specs": [spec],
            },
        )
        histories = data["project"]["run"]["sampledHistory"]
        rows = histories[0] if histories else []
        for row in rows:
            timestamp = row.get("_timestamp")
            task_id = row.get(f"task_id.{uid}")
            if (
                timestamp is None
                or timestamp < since_timestamp
                or not task_id
                or row.get(f"hotkey.{uid}") != miner_hotkey
            ):
                continue
            scores[(validator_uid, task_id)] = {
                "validator_uid": validator_uid,
                "wandb_run": run_name,
                "task_id": task_id,
                "adjusted_score": row.get(f"adjusted_score.{uid}"),
                "final_score": row.get(f"final_miner_score.{uid}"),
                "score_timestamp": timestamp,
            }
    return sorted(scores.values(), key=lambda row: row["score_timestamp"])
