from openai import OpenAI

from app.llm.base import LLMProvider


class HFInferenceLLM(LLMProvider):
    """Open instruct model via Hugging Face Inference Providers, using its OpenAI-compatible router."""

    def __init__(self, model: str, hf_token: str, base_url: str, provider: str | None = None, timeout: float = 30.0) -> None:
        self._client = OpenAI(base_url=base_url, api_key=hf_token, timeout=timeout)
        self._model = f"{model}:{provider}" if provider else model

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
