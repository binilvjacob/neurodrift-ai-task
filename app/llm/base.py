from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str: ...
