"""
Google Gemini Integration Service for StudentAI.
Used for dataset insights, alternative analysis, student improvement recommendations,
statistical interpretation, chart interpretation, and AI feedback.
Never hardcodes GEMINI_API_KEY and never invents numerical predictions.
"""

from typing import Dict, Any
import requests

from services.supabase_service import _get_config_value, _sanitize_error


class GeminiService:
    """Provides Google Gemini LLM explanations grounded in real ML and dataset statistics."""

    def __init__(self) -> None:
        self.api_key: str = _get_config_value("GEMINI_API_KEY")
        self.model: str = _get_config_value("GEMINI_MODEL", "gemini-2.5-flash")

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
        Calls Google Gemini generateContent API.
        Raises RuntimeError with sanitized error text on failure so AI Router can fallback.
        """
        if not self.is_configured():
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        gen_config: Dict[str, Any] = {"temperature": temperature}
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_config,
        }

        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API returned status {resp.status_code}: {_sanitize_error(resp.text[:200])}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini API returned no candidates.")
            content = candidates[0]["content"]["parts"][0]["text"]
            return {
                "provider": "Google Gemini",
                "model": self.model,
                "content": content,
            }
        except Exception as exc:
            raise RuntimeError(f"Gemini service error: {_sanitize_error(exc)}") from exc
