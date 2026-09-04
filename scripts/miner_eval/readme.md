# SN33 UID score analyzer

This script reads public ReadyAI validator logs from Weights & Biases and
summarizes the scores assigned to one SN33 miner UID.

## Setup

From the repository root, install the project dependencies:

```console
pip install -r requirements.txt
```

## Usage

```console
python scripts/miner_eval/miner_eval.py --uid 73
```

The default lookback is four hours. Override it when needed:

```console
python scripts/miner_eval/miner_eval.py --uid 73 --hours 24
```

The report includes final and adjusted score statistics, penalty frequency,
mean penalty size, per-validator averages, and separate averages for every
hotkey observed at the UID. The hotkey split is important when a UID was
deregistered and assigned to another miner during the selected window.

The W&B project is public, so no W&B API key is required.
