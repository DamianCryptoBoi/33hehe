from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neurons.miner import Miner


def _miner_with_tracker():
    miner = Miner.__new__(Miner)
    miner.request_tracker = MagicMock()
    miner.request_tracker.start_request.return_value = "request-id"
    miner.metagraph = type("Metagraph", (), {"hotkeys": ["validator-hotkey"]})()
    miner.wallet = type(
        "Wallet",
        (),
        {"hotkey": type("Hotkey", (), {"ss58_address": "miner-hotkey"})()},
    )()
    miner.uid = 36
    return miner


def _synapse():
    synapse = MagicMock()
    synapse.cgp_input = [{"task": {"type": "conversation_tagging", "input": {}}}]
    synapse.dendrite.hotkey = "validator-hotkey"
    return synapse


@pytest.mark.asyncio
async def test_forward_tracks_request_and_answer():
    miner = _miner_with_tracker()
    synapse = _synapse()
    task = type("Task", (), {"type": "conversation_tagging", "timeout": 12})()
    mining = AsyncMock(return_value={"tags": ["greeting", "conversation"]})

    with patch("neurons.miner.parse_task", return_value=task), patch(
        "neurons.miner.MinerLib", return_value=type("ML", (), {"do_mining": mining})()
    ):
        result = await miner.forward(synapse)

    miner.request_tracker.start_request.assert_called_once()
    start_args = miner.request_tracker.start_request.call_args.kwargs
    assert start_args["validator_hotkey"] == "validator-hotkey"
    assert start_args["validator_uid"] == 0
    assert start_args["miner_hotkey"] == "miner-hotkey"
    assert start_args["miner_uid"] == 36
    assert start_args["task_type"] == "conversation_tagging"
    assert start_args["task_payload"] == synapse.cgp_input[0]["task"]
    miner.request_tracker.finish_request.assert_called_once_with(
        "request-id",
        answer={"tags": ["greeting", "conversation"]},
        status="success",
    )
    assert result.cgp_output == [{"tags": ["greeting", "conversation"]}]


@pytest.mark.asyncio
async def test_forward_tracks_parse_errors_and_returns_empty_result():
    miner = _miner_with_tracker()
    synapse = _synapse()

    with patch("neurons.miner.parse_task", side_effect=ValueError("bad task")):
        result = await miner.forward(synapse)

    finish_args = miner.request_tracker.finish_request.call_args
    assert finish_args.args == ("request-id",)
    assert finish_args.kwargs["answer"] == {}
    assert finish_args.kwargs["status"] == "error"
    assert finish_args.kwargs["error"] == "bad task"
    assert result.cgp_output == [{}]


@pytest.mark.asyncio
async def test_forward_tracks_empty_cgp_input_and_returns_empty_result():
    miner = _miner_with_tracker()
    synapse = _synapse()
    synapse.cgp_input = []

    result = await miner.forward(synapse)

    start_args = miner.request_tracker.start_request.call_args.kwargs
    assert start_args["task_payload"] == []
    finish_args = miner.request_tracker.finish_request.call_args
    assert finish_args.kwargs["status"] == "error"
    assert result.cgp_output == [{}]


@pytest.mark.asyncio
async def test_tracker_failures_never_break_successful_mining():
    miner = _miner_with_tracker()
    miner.request_tracker.start_request.side_effect = OSError("database unavailable")
    synapse = _synapse()
    task = type("Task", (), {"type": "conversation_tagging", "timeout": 12})()
    mining = AsyncMock(return_value={"tags": ["greeting", "conversation"]})

    with patch("neurons.miner.parse_task", return_value=task), patch(
        "neurons.miner.MinerLib", return_value=type("ML", (), {"do_mining": mining})()
    ):
        result = await miner.forward(synapse)

    assert result.cgp_output == [{"tags": ["greeting", "conversation"]}]
    miner.request_tracker.finish_request.assert_not_called()


@pytest.mark.asyncio
async def test_tracker_finish_failure_never_breaks_successful_mining():
    miner = _miner_with_tracker()
    miner.request_tracker.finish_request.side_effect = OSError("database unavailable")
    synapse = _synapse()
    task = type("Task", (), {"type": "conversation_tagging", "timeout": 12})()
    mining = AsyncMock(return_value={"tags": ["greeting", "conversation"]})

    with patch("neurons.miner.parse_task", return_value=task), patch(
        "neurons.miner.MinerLib", return_value=type("ML", (), {"do_mining": mining})()
    ):
        result = await miner.forward(synapse)

    assert result.cgp_output == [{"tags": ["greeting", "conversation"]}]
