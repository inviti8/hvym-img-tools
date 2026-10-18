"""tts — a line of dialog spoken by Chatterbox, optionally in a cloned voice."""
from ...core.registry import register
from .tool import TtsInput, TtsTool

register(TtsTool)

__all__ = ["TtsTool", "TtsInput"]
