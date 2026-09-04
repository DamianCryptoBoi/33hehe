#!/usr/bin/env python3
import argparse
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

# Bittensor inspects argv while conversationgenome is imported. Keep this
# script's flags for argparse below instead of letting Bittensor consume them.
_ORIGINAL_ARGV = sys.argv[:]
try:
    sys.argv = [sys.argv[0]]
    from conversationgenome.miner.tracking import fetch_wandb_score_rows
finally:
    sys.argv = _ORIGINAL_ARGV


def analyze_scores(rows):
    valid_rows = []
    for row in rows:
        try:
            final_score = float(row["final_score"])
            adjusted_score = float(row["adjusted_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(final_score) and math.isfinite(adjusted_score)):
            continue
        valid_rows.append((row, final_score, adjusted_score))

    if not valid_rows:
        return None

    final_scores = [item[1] for item in valid_rows]
    adjusted_scores = [item[2] for item in valid_rows]
    penalties = [
        adjusted - final
        for _, final, adjusted in valid_rows
        if final < adjusted
    ]
    validator_scores = defaultdict(list)
    hotkey_scores = defaultdict(list)
    for row, final, _ in valid_rows:
        validator_scores[row["validator_uid"]].append(final)
        hotkey = row.get("miner_hotkey") or "unknown"
        hotkey_scores[hotkey].append(final)

    return {
        "total_scores": len(valid_rows),
        "mean_final_score": statistics.fmean(final_scores),
        "median_final_score": statistics.median(final_scores),
        "min_final_score": min(final_scores),
        "max_final_score": max(final_scores),
        "mean_adjusted_score": statistics.fmean(adjusted_scores),
        "penalty_count": len(penalties),
        "penalty_percentage": len(penalties) / len(valid_rows) * 100,
        "mean_penalty": statistics.fmean(penalties) if penalties else 0.0,
        "hotkey_stats": {
            hotkey: {
                "count": len(scores),
                "mean_final_score": statistics.fmean(scores),
            }
            for hotkey, scores in hotkey_scores.items()
        },
        "validator_stats": {
            uid: {
                "count": len(scores),
                "mean_final_score": statistics.fmean(scores),
            }
            for uid, scores in sorted(validator_scores.items())
        },
    }


def _hours_text(hours):
    return f"{hours:g}"


def print_report(uid, hours, stats):
    print(f"SN33 validator score analysis for UID {uid} ({_hours_text(hours)} hours)")
    print()
    print("Hotkeys observed:")
    for hotkey, hotkey_stats in sorted(
        stats["hotkey_stats"].items(),
        key=lambda item: item[1]["count"],
        reverse=True,
    ):
        print(
            f'  {hotkey}: {hotkey_stats["mean_final_score"]:.4f} '
            f'({hotkey_stats["count"]} scores)'
        )

    print()
    print(f'Total scores:          {stats["total_scores"]}')
    print(f'Mean final score:      {stats["mean_final_score"]:.4f}')
    print(f'Median final score:    {stats["median_final_score"]:.4f}')
    print(
        f'Min / max final score: {stats["min_final_score"]:.4f} / '
        f'{stats["max_final_score"]:.4f}'
    )
    print(f'Mean adjusted score:   {stats["mean_adjusted_score"]:.4f}')
    print(
        f'Penalized scores:      {stats["penalty_count"]} '
        f'({stats["penalty_percentage"]:.1f}%)'
    )
    print(f'Mean penalty:          {stats["mean_penalty"]:.4f}')

    print()
    print("Validator averages:")
    for validator_uid, validator in stats["validator_stats"].items():
        print(
            f'  UID {validator_uid}: {validator["mean_final_score"]:.4f} '
            f'({validator["count"]} scores)'
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze recent public SN33 validator scores for a miner UID."
    )
    parser.add_argument("--uid", required=True, type=int, help="SN33 miner UID")
    parser.add_argument(
        "--hours", type=float, default=4.0, help="Lookback window (default: 4)"
    )
    parser.add_argument("--entity", default="afterparty", help=argparse.SUPPRESS)
    parser.add_argument(
        "--project", default="conversationgenome", help=argparse.SUPPRESS
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.uid < 0:
        parser.error("--uid must be non-negative")
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be greater than zero")

    try:
        rows = fetch_wandb_score_rows(
            miner_uid=args.uid,
            miner_hotkey=None,
            validator_uids=None,
            since_timestamp=time.time() - args.hours * 3600,
            entity=args.entity,
            project=args.project,
        )
    except Exception as error:
        print(f"Could not fetch public W&B scores: {error}", file=sys.stderr)
        return 1

    if not rows:
        print(
            f"No validator scores found for UID {args.uid} in the last "
            f"{_hours_text(args.hours)} hours."
        )
        return 0

    stats = analyze_scores(rows)
    if stats is None:
        print(f"UID {args.uid} had score rows, but none contained valid numbers.")
        return 0

    print_report(args.uid, args.hours, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
