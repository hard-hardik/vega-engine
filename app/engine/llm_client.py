"""
Async LLM connector supporting OpenAI, Anthropic, and Groq
Includes 30s timeout guard and retry logic
"""

import asyncio
import json
import os
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    """Async LLM client with multi-provider support and timeout guards"""

    TIMEOUT_SECONDS = 25
    MAX_RETRIES = 2

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self.api_key = api_key or self._get_api_key()
        self.model = model or self._get_default_model()
        self._client: Optional[httpx.AsyncClient] = None

    def _get_api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "groq":
            return os.getenv("GROQ_API_KEY", "")
        return ""

    def _get_default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "groq": "llama-3.1-70b-versatile",
        }
        return os.getenv("LLM_MODEL", defaults.get(self.provider, "gpt-4o"))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Complete a prompt using the configured LLM provider"""
        try:
            if self.provider == "openai":
                return await self._complete_openai(prompt, system, temperature, max_tokens)
            elif self.provider == "anthropic":
                return await self._complete_anthropic(prompt, system, temperature, max_tokens)
            elif self.provider == "groq":
                return await self._complete_groq(prompt, system, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM request timed out after {self.TIMEOUT_SECONDS}s")

    async def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = await self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = await self._get_client()
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    async def _complete_groq(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = await self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def is_configured(self) -> bool:
        """Check if the LLM client is properly configured"""
        return bool(self.api_key)
