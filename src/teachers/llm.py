"""Thin wrapper over the OpenAI Python client for teacher model access.

Handles client creation and raw API calls with logprobs.
Knows nothing about classification, prompts, or datasets.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, OpenAI
from openai.types.chat import ChatCompletion


@dataclass
class TeacherConfig:
    """Connection details for a single teacher LLM.

    All providers use OpenAI-compatible APIs, so the only
    differences are base_url and api_key.

    extra_body holds provider-specific request fields. For OpenRouter we set
    provider.require_parameters=True so requests are only routed to backends
    that actually support logprobs/top_logprobs (otherwise logprobs come back
    None and parsing fails).
    """

    name: str
    model: str
    base_url: str
    api_key_env: str
    max_top_logprobs: int = 20
    max_output_tokens: int = 5
    use_max_completion_tokens: bool = False
    constrain_to_verbalizers: bool = False
    extra_body: dict = field(default_factory=dict)


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_REQUIRE_LOGPROBS = {"provider": {"require_parameters": True}}
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
# Gemma 4 / Qwen 3.5 / gpt-oss enable thinking by default on Ollama; that emits
# <|channel|>/thought before the answer and breaks first-token logit parsing.
_OLLAMA_EXTRA_BODY = {"reasoning_effort": "none"}
# Gemma 4 on vLLM: keep thinking off so the first output token is the class letter.
_VLLM_GEMMA4_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
# Qwen on vLLM: keep thinking off for first-token logprob extraction.
_VLLM_QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


TEACHER_PRESETS: dict[str, TeacherConfig] = {
    "gpt-4.1-mini": TeacherConfig(
        name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        base_url=_OPENAI_BASE_URL,
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt-4o": TeacherConfig(
        name="gpt-4o",
        model="gpt-4o",
        base_url=_OPENAI_BASE_URL,
        api_key_env="OPENAI_API_KEY",
    ),
    "gemma-4-31b": TeacherConfig(
        name="gemma-4-31b",
        model="google/gemma-4-31b-it",
        base_url=_OPENROUTER_BASE_URL,
        api_key_env="OPENROUTER_API_KEY",
        extra_body=_OPENROUTER_REQUIRE_LOGPROBS,
    ),
    # Self-hosted via RunPod serverless vLLM endpoint.
    # Set RUNPOD_API_KEY and update `model` to match what vLLM loaded
    # (check: GET https://api.runpod.ai/v2/65ysfjx24c641r/openai/v1/models).
    "llama-runpod": TeacherConfig(
        name="llama-runpod",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # update if different
        base_url="https://api.runpod.ai/v2/65ysfjx24c641r/openai/v1",
        api_key_env="RUNPOD_API_KEY",
    ),
    # Local Ollama (OpenAI-compatible endpoint). Run `ollama serve` and pull the
    # model first, e.g. `ollama pull gemma4:31b-mlx`. Set OLLAMA_API_KEY=ollama.
    "gemma4-31b-mlx": TeacherConfig(
        name="gemma4-31b-mlx",
        model="gemma4:31b-mlx",
        base_url=_OLLAMA_BASE_URL,
        api_key_env="OLLAMA_API_KEY",
        extra_body=_OLLAMA_EXTRA_BODY,
    ),
    "gemma4-26b-mlx": TeacherConfig(
        name="gemma4-26b-mlx",
        model="gemma4:26b-mlx",
        base_url=_OLLAMA_BASE_URL,
        api_key_env="OLLAMA_API_KEY",
        extra_body=_OLLAMA_EXTRA_BODY,
    ),
    # Local vLLM (OpenAI-compatible). Start with ./start_vllm.sh first.
    # Server must be launched with --max-logprobs 262144 (see start_vllm.sh).
    "gemma4-26b-vllm": TeacherConfig(
        name="gemma4-26b-vllm",
        model="google/gemma-4-26B-A4B-it",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=50,
        extra_body=_VLLM_GEMMA4_EXTRA_BODY,
    ),
    "gemma4-26b-vllm-guided": TeacherConfig(
        name="gemma4-26b-vllm-guided",
        model="google/gemma-4-26B-A4B-it",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=20,
        constrain_to_verbalizers=True,
        extra_body=_VLLM_GEMMA4_EXTRA_BODY,
    ),
    "qwen3.5-27b-vllm": TeacherConfig(
        name="qwen3.5-27b-vllm",
        model="Qwen/Qwen3.5-27B",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=50,
        extra_body=_VLLM_QWEN_EXTRA_BODY,
    ),
    "qwen3.5-27b-vllm-guided": TeacherConfig(
        name="qwen3.5-27b-vllm-guided",
        model="Qwen/Qwen3.5-27B",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=20,
        constrain_to_verbalizers=True,
        extra_body=_VLLM_QWEN_EXTRA_BODY,
    ),
    "mistral-small-24b-vllm": TeacherConfig(
        name="mistral-small-24b-vllm",
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=50,
    ),
    "mistral-small-24b-vllm-guided": TeacherConfig(
        name="mistral-small-24b-vllm-guided",
        model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        base_url=_VLLM_BASE_URL,
        api_key_env="VLLM_API_KEY",
        max_top_logprobs=20,
        constrain_to_verbalizers=True,
    ),
    "qwen3.5-27b": TeacherConfig(
        name="qwen3.5-27b",
        model="qwen3.5:27b",
        base_url=_OLLAMA_BASE_URL,
        api_key_env="OLLAMA_API_KEY",
        extra_body=_OLLAMA_EXTRA_BODY,
    ),
}


def has_api_credentials(config: TeacherConfig) -> bool:
    """Return True if ``get_client`` can run without raising."""
    if os.environ.get(config.api_key_env):
        return True
    if config.base_url == _OLLAMA_BASE_URL:
        return True
    if config.base_url == _VLLM_BASE_URL:
        return True
    return False


def get_client(config: TeacherConfig) -> OpenAI:
    """Create an OpenAI client from a TeacherConfig.

    Reads the API key from the environment variable specified
    in ``config.api_key_env``.
    """
    api_key = os.environ.get(config.api_key_env)
    if not api_key and config.base_url == _OLLAMA_BASE_URL:
        api_key = "ollama"
    if not api_key and config.base_url == _VLLM_BASE_URL:
        api_key = "mysecretkey"
    if not api_key:
        raise RuntimeError(
            f"Environment variable {config.api_key_env} is not set. "
            f"Set it to the API key for {config.name}."
        )
    base_url = config.base_url
    if base_url == _OPENAI_BASE_URL:
        base_url = os.environ.get("OPENAI_BASE_URL", _OPENAI_BASE_URL)
    if base_url == _VLLM_BASE_URL:
        base_url = os.environ.get("VLLM_BASE_URL", _VLLM_BASE_URL)
    return OpenAI(base_url=base_url, api_key=api_key)


def is_local_vllm(config: TeacherConfig) -> bool:
    """Return True for the local vLLM teacher preset."""
    return config.base_url == _VLLM_BASE_URL


def wait_until_ready(
    client: OpenAI,
    config: TeacherConfig,
    timeout_s: int = 900,
    interval_s: int = 10,
) -> None:
    """Wait for a local OpenAI-compatible server to accept requests."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            models = client.models.list()
            model_ids = {model.id for model in models.data}
            if is_local_vllm(config) and config.model not in model_ids:
                raise RuntimeError(
                    f"{config.name} expects vLLM to serve {config.model!r}, "
                    f"but the server currently exposes {sorted(model_ids)!r}. "
                    "Update VLLM_MODEL in .env and restart ./start_vllm.sh."
                )
            return
        except APIConnectionError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{config.name} is not reachable at {config.base_url}. "
                    "Start vLLM with ./start_vllm.sh and wait for "
                    "'Application startup complete' in vllm.log."
                ) from exc
            print(
                f"Waiting for {config.name} server to become ready "
                f"({config.base_url})..."
            )
            time.sleep(interval_s)


def complete(
    client: OpenAI,
    config: TeacherConfig,
    messages: list[dict[str, str]],
    guided_choices: list[str] | None = None,
) -> ChatCompletion:
    """Call chat completions with logprobs enabled.

    Returns the raw ChatCompletion response object.
    """
    token_kw = (
        {"max_completion_tokens": config.max_output_tokens}
        if config.use_max_completion_tokens
        else {"max_tokens": config.max_output_tokens}
    )
    extra_body = dict(config.extra_body)
    if guided_choices is not None:
        # vLLM 0.23 uses structured_outputs; legacy guided_choice is ignored.
        extra_body["structured_outputs"] = {"choice": guided_choices}

    return client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=0,
        logprobs=True,
        top_logprobs=config.max_top_logprobs,
        extra_body=extra_body,
        **token_kw,
    )
