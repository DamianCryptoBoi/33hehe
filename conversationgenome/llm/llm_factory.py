import json
from pathlib import Path

import bittensor as bt

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib

LOCKED_LLM_TYPE = "openai"
DEFAULT_MINER_LLM_CONFIG_PATH = Path(__file__).resolve().parents[2] / "miner_llm_config.json"
MINER_TASK_TYPES = {
    "conversation_tagging",
    "webpage_metadata_generation",
    "survey_tagging",
    "named_entities_extraction",
    "skill_generation",
    "skill_coverage_evaluation",
}
SUPPORTED_LLM_TYPES = {"openai", "anthropic", "groq", "openrouter", "chutes", "vertex"}
ROUTE_FIELDS = {
    "openai": {"provider", "model", "base_url", "reasoning_effort"},
    "vertex": {"provider", "model", "reasoning_effort"},
}
REASONING_EFFORTS = {
    "openai": {"none", "low", "medium", "high", "xhigh", "max"},
    "vertex": {"low", "medium", "high"},
}
_OVERRIDE_ENV_VARS = [
    "LLM_TYPE_OVERRIDE",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDINGS_MODEL_OVERRIDE",
]


def _present_llm_override_vars() -> list[str]:
    return [v for v in _OVERRIDE_ENV_VARS if c.get("env", v)]


def configure_llm_override_lockdown(netuid: int) -> bool:
    """Validator-only startup check: on mainnet, LLM provider/model/embeddings-model
    overrides from .env are ignored so all mainnet validators score with the same
    in-code default LLM config. On any other netuid, overrides are still honored but
    a warning is logged so operators notice before running on mainnet.
    """
    mainnet_netuid = c.get("network", "mainnet", 33)
    is_mainnet = netuid == mainnet_netuid
    c.set("system", "llm_overrides_locked", is_mainnet)

    present = _present_llm_override_vars()
    if present:
        if is_mainnet:
            bt.logging.warning(
                f"LLM override env var(s) {present} detected on mainnet (netuid={netuid}). "
                "Overrides are ignored on mainnet; falling back to in-code defaults."
            )
        else:
            bt.logging.warning(
                f"LLM override env var(s) {present} detected on non-mainnet netuid={netuid}. "
                "Honoring overrides now, but they will be ignored if this validator runs on mainnet."
            )
    return is_mainnet


def _validate_miner_route(name: str, route: object) -> dict:
    if not isinstance(route, dict):
        raise ValueError(f"Miner LLM route '{name}' must be an object")

    provider = route.get("provider")
    model = route.get("model")
    if provider not in SUPPORTED_LLM_TYPES:
        raise ValueError(f"Miner LLM route '{name}' has unsupported provider: {provider}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Miner LLM route '{name}' requires a model")

    allowed_fields = ROUTE_FIELDS.get(provider, {"provider", "model"})
    unknown_fields = set(route) - allowed_fields
    if unknown_fields:
        raise ValueError(
            f"Miner LLM route '{name}' has unknown field(s): "
            f"{', '.join(sorted(unknown_fields))}"
        )

    reasoning_effort = route.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
    ):
        raise ValueError(f"Miner LLM route '{name}' has invalid reasoning_effort")
    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.lower()
        if reasoning_effort not in REASONING_EFFORTS[provider]:
            raise ValueError(
                f"Miner LLM route '{name}' has unsupported reasoning_effort: "
                f"{reasoning_effort}"
            )

    base_url = route.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ValueError(f"Miner LLM route '{name}' has invalid base_url")

    validated = dict(route)
    if reasoning_effort is not None:
        validated["reasoning_effort"] = reasoning_effort
    return validated


def load_miner_llm_config() -> dict:
    config_path = Path(
        c.get("env", "MINER_LLM_CONFIG", str(DEFAULT_MINER_LLM_CONFIG_PATH))
    ).expanduser()
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load miner LLM config {config_path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError("Miner LLM config must be an object")

    unknown_fields = set(config) - {"default", "tasks"}
    if unknown_fields:
        raise ValueError(
            "Miner LLM config has unknown field(s): "
            f"{', '.join(sorted(unknown_fields))}"
        )

    default = _validate_miner_route("default", config.get("default"))
    tasks = config.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("Miner LLM config 'tasks' must be an object")

    unknown_tasks = set(tasks) - MINER_TASK_TYPES
    if unknown_tasks:
        raise ValueError(f"Unknown miner task type(s): {', '.join(sorted(unknown_tasks))}")

    return {
        "default": default,
        "tasks": {
            task_type: _validate_miner_route(task_type, route)
            for task_type, route in tasks.items()
        },
    }


def _create_llm_backend(llm_type: str, request_timeout=None, route=None) -> LlmLib:
    route = route or {}
    model = route.get("model")
    reasoning_effort = route.get("reasoning_effort")

    if llm_type == "openai":
        from .llm_openai import LlmOpenAI
        return LlmOpenAI(
            request_timeout=request_timeout,
            model=model,
            base_url=route.get("base_url"),
            reasoning_effort=reasoning_effort,
        )
    if llm_type == "anthropic":
        from .llm_anthropic import LlmAnthropic
        backend = LlmAnthropic()
    elif llm_type == "groq":
        from .llm_groq import LlmGroq
        backend = LlmGroq()
    elif llm_type == "openrouter":
        from .llm_openrouter import LlmOpenRouter
        backend = LlmOpenRouter()
    elif llm_type == "chutes":
        from .llm_chutes import LlmChutes
        backend = LlmChutes()
    elif llm_type == "vertex":
        from .llm_vertex import LlmVertex
        return LlmVertex(
            request_timeout=request_timeout,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {llm_type}")

    if model:
        backend.model = model
    if reasoning_effort:
        backend.reasoning_effort = reasoning_effort
    return backend


def check_miner_llm_backends(request_timeout: float = 10) -> None:
    config = load_miner_llm_config()
    routes = [("default", config["default"]), *config["tasks"].items()]
    checked = set()

    for name, route in routes:
        fingerprint = (
            route["provider"],
            route["model"],
            route.get("base_url"),
            route.get("reasoning_effort"),
        )
        if fingerprint in checked:
            continue
        checked.add(fingerprint)

        try:
            backend = _create_llm_backend(
                route["provider"], request_timeout=request_timeout, route=route
            )
            response = backend.basic_prompt("Reply with exactly OK.")
        except Exception as error:
            raise RuntimeError(
                f"Miner LLM startup check failed for {name}: "
                f"{route['provider']}/{route['model']}: {error}"
            ) from error
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(
                f"Miner LLM startup check failed for {name}: "
                f"{route['provider']}/{route['model']} returned no response"
            )

        bt.logging.info(
            f"Miner LLM startup check passed for {name}: "
            f"{route['provider']}/{route['model']}"
        )


def get_llm_backend(llm_type_override=None, request_timeout=None, task_type=None) -> LlmLib:
    """Return a task-routed miner backend or the legacy global backend."""
    if c.get("system", "llm_overrides_locked", False):
        from .llm_openai import LlmOpenAI
        return LlmOpenAI(ignore_model_override=True, request_timeout=request_timeout)

    route = None
    if task_type is not None:
        if task_type not in MINER_TASK_TYPES:
            raise ValueError(f"Unknown miner task type: {task_type}")
        config = load_miner_llm_config()
        route = config["tasks"].get(task_type, config["default"])
        llm_type_override = route["provider"]

    if not llm_type_override:
        llm_type_override = c.get("env", "LLM_TYPE_OVERRIDE")

    return _create_llm_backend(
        llm_type_override or "openai",
        request_timeout=request_timeout,
        route=route,
    )
