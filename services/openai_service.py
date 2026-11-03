"""
OpenAI Integration Service for StudentAI.
Used for natural-language prediction explanations, academic recommendations,
executive summaries, AI-generated reports, and feature importance explanations.
Never hardcodes OPENAI_API_KEY and never invents numerical predictions.
"""

import json
from typing import Dict, Any, Optional
import requests

from services.supabase_service import _get_config_value, _sanitize_error


class OpenAIService:
    """Provides OpenAI LLM explanations strictly grounded in actual ML and dataset outputs."""

    def __init__(self) -> None:
        self.api_key: str = _get_config_value("OPENAI_API_KEY")
        self.model: str = _get_config_value("OPENAI_MODEL", "gpt-4o-mini")

    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("YOUR_"))

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Executes a single request to the OpenAI Chat Completions API.
        Raises RuntimeError with sanitized error text on failure so AI Router can fallback.
        """
        if not self.is_configured():
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API returned status {resp.status_code}: {_sanitize_error(resp.text[:200])}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "provider": "OpenAI",
                "model": self.model,
                "content": content,
            }
        except Exception as exc:
            raise RuntimeError(f"OpenAI service error: {_sanitize_error(exc)}") from exc
