"""LLM provider layer: Ollama for local development, Groq for hosted deployments."""

import logging
from collections.abc import Iterator
from typing import cast

import groq
import ollama
from config import settings
from groq.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "The language model is unavailable right now. Please try again shortly."
_PROVIDER_ERRORS = (ollama.ResponseError, ConnectionError, groq.GroqError)


class LLMError(RuntimeError):
    """The configured LLM provider is misconfigured or failed to respond."""


def active_model() -> str:
    return settings.groq_model if settings.llm_provider == "groq" else settings.ollama_model


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_base_url)


def _groq_client() -> groq.Groq:
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    if not api_key:
        raise LLMError("LLM_PROVIDER is 'groq' but GROQ_API_KEY is not set")
    return groq.Groq(api_key=api_key)


def chat(messages: list[dict]) -> str:
    try:
        if settings.llm_provider == "groq":
            completion = _groq_client().chat.completions.create(
                model=settings.groq_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
            )
            return completion.choices[0].message.content or ""
        response = _ollama_client().chat(model=settings.ollama_model, messages=messages)
        return str(response["message"]["content"])
    except _PROVIDER_ERRORS as e:
        logger.exception("LLM request failed (provider=%s)", settings.llm_provider)
        raise LLMError(UNAVAILABLE_MESSAGE) from e


def stream_chat(messages: list[dict]) -> Iterator[str]:
    try:
        if settings.llm_provider == "groq":
            stream = _groq_client().chat.completions.create(
                model=settings.groq_model,
                messages=cast("list[ChatCompletionMessageParam]", messages),
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
            return
        for part in _ollama_client().chat(
            model=settings.ollama_model, messages=messages, stream=True
        ):
            yield str(part["message"]["content"])
    except _PROVIDER_ERRORS as e:
        logger.exception("LLM stream failed (provider=%s)", settings.llm_provider)
        raise LLMError(UNAVAILABLE_MESSAGE) from e
