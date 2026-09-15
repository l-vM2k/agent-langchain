"""LLM factory — provider routing, mirroring AgentConfigResolver.createAgentscopeModel."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from xiaowei.config import ModelConfig


def create_llm(cfg: ModelConfig) -> BaseChatModel:
    """Build a LangChain chat model from a ModelConfig.

    Routing mirrors the Java side:
      - XIAOWEI_FAKE_LLM=1            -> FakeChatModel (offline, no network)
      - provider_code == "ollama"    -> ChatOllama
      - anything else (qianwen/deepseek/openai/custom_api...)
                                     -> ChatOpenAI (OpenAI-compatible)
    """
    if os.environ.get("XIAOWEI_FAKE_LLM") == "1":
        from xiaowei.llm.fake import FakeChatModel

        return FakeChatModel()

    if cfg.provider_code == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=cfg.api_host,
            model=cfg.model_name,
            temperature=cfg.temperature,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=cfg.api_host,
        api_key=cfg.api_key,
        model=cfg.model_name,
        temperature=cfg.temperature,
        streaming=cfg.stream,
        extra_body=cfg.extra_body or None,
    )
