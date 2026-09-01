#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

# Bittensor's eager package imports inspect sys.argv and otherwise consume this
# script's --help flag before argparse gets it.
_ORIGINAL_ARGV = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    from conversationgenome.miner.tracking import RequestTracker
    from conversationgenome.miner.tracking import fetch_wandb_score_rows
    from conversationgenome.miner.tracking import match_score_rows
finally:
    sys.argv = _ORIGINAL_ARGV


DEFAULT_DB = str(Path.home() / ".bittensor" / "miner_requests.sqlite3")


def _format_time(timestamp):
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _print_rows(rows):
    print(
        "received             validator task                         "
        "ms     status   score    confidence request_id"
    )
    for row in rows:
        score = "-" if row["final_score"] is None else f'{row["final_score"]:.4f}'
        duration = "-" if row["duration_ms"] is None else str(row["duration_ms"])
        print(
            f'{_format_time(row["received_at"]):19} '
            f'{str(row["validator_uid"]):9} '
            f'{(row["task_type"] or "-")[:28]:28} '
            f'{duration:6} '
            f'{row["status"][:8]:8} '
            f'{score:8} '
            f'{(row["match_confidence"] or "-"):10} '
            f'{row["request_id"]}'
        )


def _show(tracker, request_id):
    row = tracker.get_request(request_id)
    if row is None:
        print(f"Request not found: {request_id}", file=sys.stderr)
        return 1
    for field in ("task_payload", "answer"):
        if row[field]:
            row[field] = json.loads(row[field])
    print(json.dumps(row, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _sync(tracker, args):
    since = time.time() - (args.hours * 3600)
    requests = tracker.pending_requests(
        miner_hotkey=args.miner_hotkey,
        miner_uid=args.miner_uid,
        since_timestamp=since,
    )
    if not requests:
        print("No pending tracked requests in the selected window.")
        return 0

    validator_uids = {
        row["validator_uid"] for row in requests if row["validator_uid"] is not None
    }
    scores = fetch_wandb_score_rows(
        miner_hotkey=args.miner_hotkey,
        miner_uid=args.miner_uid,
        validator_uids=validator_uids,
        since_timestamp=min(row["received_at"] for row in requests),
        entity=args.entity,
        project=args.project,
    )
    matches = match_score_rows(
        requests,
        scores,
        max_delay_seconds=args.max_delay_seconds,
    )
    attachable = [
        match
        for match in matches
        if match["match_confidence"] == "high" or args.include_ambiguous
    ]
    for match in attachable:
        tracker.attach_score(
            match["request_id"],
            wandb_run=match["wandb_run"],
            task_id=match["task_id"],
            adjusted_score=match["adjusted_score"],
            final_score=match["final_score"],
            score_timestamp=match["score_timestamp"],
            match_confidence=match["match_confidence"],
        )
    skipped = len(matches) - len(attachable)
    print(
        f"Attached {len(attachable)} of {len(requests)} pending requests; "
        f"skipped {skipped} ambiguous matches."
    )
    _print_rows(tracker.recent_requests(limit=args.limit))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect tracked SN33 miner requests and reconcile their public W&B scores."
    )
    parser.add_argument(
        "--db",
        default=os.getenv("MINER_TRACKING_DB", DEFAULT_DB),
        help="SQLite tracking database path",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List recent tracked requests")
    list_parser.add_argument("--limit", type=int, default=20)

    show_parser = commands.add_parser("show", help="Show one request and its answer")
    show_parser.add_argument("request_id")

    sync_parser = commands.add_parser("sync", help="Fetch and attach public W&B scores")
    sync_parser.add_argument("--miner-hotkey", required=True)
    sync_parser.add_argument("--miner-uid", required=True, type=int)
    sync_parser.add_argument("--hours", type=float, default=72)
    sync_parser.add_argument("--max-delay-seconds", type=float, default=900)
    sync_parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Attach medium-confidence time/order matches",
    )
    sync_parser.add_argument("--limit", type=int, default=20)
    sync_parser.add_argument("--entity", default="afterparty")
    sync_parser.add_argument("--project", default="conversationgenome")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tracker = RequestTracker(args.db)
    if args.command == "list":
        _print_rows(tracker.recent_requests(limit=args.limit))
        return 0
    if args.command == "show":
        return _show(tracker, args.request_id)
    try:
        return _sync(tracker, args)
    except Exception as error:
        print(f"Score reconciliation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
