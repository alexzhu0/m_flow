from __future__ import annotations

from m_flow.api.v1.responses.models import ResponseRequest


def test_response_request_schema_documents_required_payload_shape() -> None:
    schema = ResponseRequest.model_json_schema()
    props = schema["properties"]

    assert schema["required"] == ["input"]
    assert "input" in props
    assert "model" in props
    assert "text" in props["input"]["description"].lower()
    assert "response model alias" in props["model"]["description"].lower()
    assert schema["examples"] == [
        {
            "model": "mflow_v1",
            "input": "Summarize the latest project update",
            "tool_choice": "auto",
        }
    ]
