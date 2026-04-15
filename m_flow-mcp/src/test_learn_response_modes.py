from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from learn_response_utils import is_api_mode_learn_response, is_direct_mode_learn_response


def test_direct_mode_learn_response_accepts_success_and_no_data_messages() -> None:
    assert is_direct_mode_learn_response("✅ 学习完成\n{}")
    assert is_direct_mode_learn_response("ℹ️ 当前没有可学习的数据")
    assert not is_direct_mode_learn_response("⚠️ API 模式不支持")


def test_api_mode_learn_response_requires_explicit_api_feedback() -> None:
    assert is_api_mode_learn_response("⚠️ API 模式不支持 learn")
    assert is_api_mode_learn_response("✅ 学习完成\n{\"success\": true}")
    assert not is_api_mode_learn_response("返回正常但没有说明模式")
