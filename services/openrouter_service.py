"""
OpenRouter Integration Service for StudentAI.
Provides flexible model access, configurable model selection via OPENROUTER_MODEL,
and reliable fallback AI explanation capabilities.
"""

from typing import Dict, Any, Optional
import requests

from services.supabase_service import _get_config_value, _sanitize_error


class OpenRouterService:
    """Provides OpenRouter API access using configurable OPENROUTER_API_KEY and OPENROUTER_MODEL."""

    def __init__(self, custom_model: Optional[str] = None) -> None:
        self.api_key: str = _get_config_value("OPENROUTER_API_KEY")
        self.model: str = custom_model or _get_config_value("OPENROUTER_MODEL", "openai/gpt-4o-mini")

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
        Calls OpenRouter chat completions endpoint.
        Raises RuntimeError with sanitized error text on failure so AI Router can handle fallback.
        """
        if not self.is_configured():
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://studentai.streamlit.app",
            "X-Title": "StudentAI Academic Performance Analyzer",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=22,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API returned status {resp.status_code}: {_sanitize_error(resp.text[:200])}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "provider": "OpenRouter",
                "model": self.model,
                "content": content,
            }
        except Exception as exc:
            raise RuntimeError(f"OpenRouter service error: {_sanitize_error(exc)}") from exc
