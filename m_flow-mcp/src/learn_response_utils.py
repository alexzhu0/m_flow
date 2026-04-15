from __future__ import annotations


def is_direct_mode_learn_response(content: str) -> bool:
    """Return True for direct-mode learn responses that indicate success or no work."""
    return "学习完成" in content or "没有可学习的数据" in content


def is_api_mode_learn_response(content: str) -> bool:
    """Return True only when API-mode learn reports explicit support or limitation text."""
    return "学习完成" in content or "API 模式" in content or "直接模式" in content
