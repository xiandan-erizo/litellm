"""
Tests for forcing the /responses → /chat/completions bridge for `openai/` models
(via `use_chat_completions_api` or the `openai/chat_completions/<model>` model id).

Includes file_search emulation: the flag must be forwarded on inner aresponses
calls so routed requests do not hit a custom api_base /v1/responses endpoint.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse


class TestUseResponsesApiBridgeFlag:
    """Test that bridge opt-in forces the chat completions path."""

    def test_responses_tool_choice_function_name_maps_to_chat_format(self):
        """Responses tool_choice uses top-level name; chat completions needs function.name."""
        assert LiteLLMCompletionResponsesConfig._transform_tool_choice(
            {"type": "function", "name": "get_weather"}
        ) == {"type": "function", "function": {"name": "get_weather"}}

    def test_chat_tool_choice_function_name_maps_to_responses_format(self):
        """Response events need Responses tool_choice with a top-level name."""
        assert LiteLLMCompletionResponsesConfig._transform_tool_choice_to_responses(
            {"type": "function", "function": {"name": "get_weather"}}
        ) == {"type": "function", "name": "get_weather"}

    def test_chat_completion_tool_format_preserved_by_bridge(self):
        """Codex-style chat completion tools should not lose function metadata."""
        tools, web_search_options = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ]
            )
        )

        assert web_search_options is None
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": False,
                },
            }
        ]

    def test_responses_only_tools_are_dropped_by_bridge(self):
        """Tools with no chat completions equivalent should not leak upstream."""
        tools, web_search_options = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                [
                    {"type": "custom", "name": "shell"},
                    {"type": "local_shell", "name": "shell"},
                    {"type": "namespace", "name": "terminal"},
                    {"type": "file_search"},
                    {"type": "image_generation"},
                    {"type": "code_interpreter"},
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {"type": "object"},
                        },
                    },
                ]
            )
        )

        assert web_search_options is None
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "read"  # type: ignore[index]

    def test_tool_output_content_parts_are_flattened_to_text(self):
        """Tool message content should stay text-only for OpenAI-compatible chat providers."""
        messages = (
            LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {"type": "input_text", "text": "screenshot saved"},
                        {
                            "type": "input_image",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                    ],
                }
            )
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"  # type: ignore[index]
        assert (  # type: ignore[index]
            messages[0]["content"]
            == "screenshot saved[image: data:image/png;base64,abc]"
        )

    def test_response_metadata_tools_preserve_responses_only_tools(self):
        """response.created metadata should reflect requested Responses tools."""
        tools = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_response_metadata_tools(
                [
                    {"type": "web_search_preview", "search_context_size": "low"},
                    {"type": "file_search", "vector_store_ids": ["vs_123"]},
                    {"type": "code_interpreter", "container": {"type": "auto"}},
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                        },
                    },
                ]
            )
        )

        assert tools[0]["type"] == "web_search_preview"
        assert tools[1]["type"] == "file_search"
        assert tools[1]["vector_store_ids"] == ["vs_123"]
        assert tools[2]["type"] == "code_interpreter"
        assert tools[3]["type"] == "function"
        assert tools[3]["name"] == "read"

    def test_responses_tool_format_maps_to_chat_format(self):
        """Responses API function tools should still map to chat completion tools."""
        tools, web_search_options = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ]
            )
        )

        assert web_search_options is None
        assert tools[0]["function"]["name"] == "get_weather"  # type: ignore[index]
        assert tools[0]["function"]["parameters"]["type"] == "object"  # type: ignore[index]

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_use_chat_completions_api_true(
        self, mock_get_config, mock_bridge_handler
    ):
        """When use_chat_completions_api=True, the bridge handler should be called."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_model_uses_chat_completions_prefix(
        self, mock_get_config, mock_bridge_handler
    ):
        """`openai/chat_completions/<name>` normalizes to `openai/<name>` and uses the bridge."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/chat_completions/my-custom-model",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()
        # Model string is provider-normalized after resolution; prefix only forces the bridge.
        assert mock_bridge_handler.call_args.kwargs["model"].endswith("my-custom-model")

    @patch("litellm.responses.main.base_llm_http_handler.response_api_handler")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_native_forwarding_when_flag_absent(
        self, mock_get_config, mock_native_handler
    ):
        """When use_chat_completions_api is not set, openai/ models should use
        native responses API forwarding (existing behavior)."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_native_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_flag_does_not_leak_into_kwargs(self, mock_get_config, mock_bridge_handler):
        """use_chat_completions_api should be popped and not passed to the bridge handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="openai/my-custom-model",
            input="Hello",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        call_kwargs = mock_bridge_handler.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "use_chat_completions_api" not in all_kwargs

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch("litellm.responses.main.base_llm_http_handler.compact_response_api_handler")
    def test_compact_responses_bridge_used_when_flag_true(
        self, mock_native_compact_handler, mock_bridge_handler
    ):
        """Compact should use the chat-completions bridge for bridge-only deployments."""
        mock_bridge_handler.return_value = ResponsesAPIResponse(
            id="resp_compact",
            model="openai/my-custom-model",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Compacted context"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )

        result = litellm.compact_responses(
            model="openai/my-custom-model",
            input=[{"role": "user", "content": "Hello"}],
            instructions="Keep project details",
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        mock_native_compact_handler.assert_not_called()
        mock_bridge_handler.assert_called_once()
        call_kwargs = mock_bridge_handler.call_args.kwargs
        assert call_kwargs["stream"] is False
        assert "use_chat_completions_api" not in call_kwargs
        assert (
            call_kwargs["responses_api_request"]["instructions"]
            .startswith("Compact the provided conversation")
        )
        assert "Keep project details" in call_kwargs["responses_api_request"][
            "instructions"
        ]
        assert result.id == "resp_compact"

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    def test_bridge_used_when_provider_config_none(
        self, mock_get_config, mock_bridge_handler
    ):
        """When the provider has no native responses API config (returns None),
        the bridge should be used regardless of the flag (existing behavior)."""
        mock_get_config.return_value = None
        mock_bridge_handler.return_value = MagicMock()

        litellm.responses(
            model="anthropic/claude-3-haiku",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_bridge_handler.assert_called_once()

    @patch("litellm.responses.file_search.emulated_handler._call_aresponses")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_bridge_flag_forwarded_to_file_search_emulation(
        self, mock_get_config, mock_call_aresponses
    ):
        """When use_chat_completions_api=True and file_search tool is present,
        the flag should be forwarded to the inner aresponses call in the
        file_search emulation path."""
        # Setup: provider has native responses API support
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()

        # Mock the inner aresponses call to return a valid response
        mock_response = ResponsesAPIResponse(
            id="resp_123",
            model="openai/my-custom-model",
            created_at=1234567890,
            output=[
                {"type": "message", "content": [{"type": "text", "text": "Answer"}]}
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        mock_call_aresponses.return_value = mock_response

        await litellm.aresponses(
            model="openai/my-custom-model",
            input="Search for information",
            tools=[{"type": "file_search"}],
            use_chat_completions_api=True,
            litellm_logging_obj=MagicMock(),
        )

        # Verify _call_aresponses was called with use_chat_completions_api=True
        mock_call_aresponses.assert_called_once()
        call_kwargs = mock_call_aresponses.call_args.kwargs
        assert (
            call_kwargs.get("use_chat_completions_api") is True
        ), "use_chat_completions_api should be forwarded to inner aresponses call"

    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    @patch("litellm.vector_stores.main.asearch")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_bridge_flag_prevents_native_responses_endpoint_call(
        self, mock_get_config, mock_asearch, mock_bridge_handler
    ):
        """
        Concrete failing scenario: native OpenAI responses config + bridge flag +
        file_search → emulation must still route inner calls through the bridge
        (chat completions), not POST to api_base /v1/responses.
        """
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []

        first_response = ResponsesAPIResponse(
            id="resp_first",
            model="openai/my-local-model",
            created_at=1234567890,
            output=[
                {
                    "type": "function_call",
                    "name": "litellm_file_search",
                    "call_id": "call_123",
                    "arguments": '{"queries": ["test query"]}',
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )
        second_response = ResponsesAPIResponse(
            id="resp_second",
            model="openai/my-local-model",
            created_at=1234567891,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Final answer"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=20, output_tokens=10, total_tokens=30
            ),
        )
        mock_bridge_handler.side_effect = [first_response, second_response]

        result = await litellm.aresponses(
            model="openai/my-local-model",
            input="Search for information",
            tools=[
                {
                    "type": "file_search",
                    "file_search": {"vector_store_ids": ["vs_123"]},
                }
            ],
            use_chat_completions_api=True,
            api_base="http://localhost:8080/v1",
            litellm_logging_obj=MagicMock(),
        )

        assert mock_bridge_handler.call_count == 2, (
            "Bridge handler should be called twice: initial function-tool call "
            "and follow-up with tool results"
        )
        for call in mock_bridge_handler.call_args_list:
            all_kwargs = call.kwargs if call.kwargs else {}
            assert "use_chat_completions_api" not in all_kwargs
        assert result is not None
        assert result.id is not None

    @patch("litellm.responses.main.base_llm_http_handler.response_api_handler")
    @patch("litellm.vector_stores.main.asearch")
    @patch(
        "litellm.responses.main.ProviderConfigManager.get_provider_responses_api_config"
    )
    async def test_without_bridge_flag_uses_native_endpoint(
        self, mock_get_config, mock_asearch, mock_native_handler
    ):
        """Without the bridge flag, openai/ with native config uses the native handler."""
        mock_get_config.return_value = litellm.OpenAIResponsesAPIConfig()
        mock_asearch.return_value = []
        mock_native_handler.return_value = ResponsesAPIResponse(
            id="resp_native",
            model="openai/gpt-4o",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Native response"}],
                }
            ],
            usage=ResponseAPIUsage(
                input_tokens=10, output_tokens=5, total_tokens=15
            ),
        )

        result = await litellm.aresponses(
            model="openai/gpt-4o",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_native_handler.assert_called_once()
        assert result is not None

    def test_reasoning_input_item_detected_correctly(self):
        """Reasoning items from OpenAI thinking models should be recognized."""
        assert LiteLLMCompletionResponsesConfig._is_input_item_reasoning(
            {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "Thinking..."}],
            }
        ) is True

    def test_non_reasoning_items_not_detected_as_reasoning(self):
        """Other input item types should not be detected as reasoning."""
        assert LiteLLMCompletionResponsesConfig._is_input_item_reasoning(
            {"type": "function_call", "call_id": "call_123"}
        ) is False
        assert LiteLLMCompletionResponsesConfig._is_input_item_reasoning(
            {"type": "message", "role": "user", "content": "Hello"}
        ) is False
        assert LiteLLMCompletionResponsesConfig._is_input_item_reasoning(
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "result",
            }
        ) is False

    def test_reasoning_input_item_transforms_to_assistant_message(self):
        """Reasoning items should become assistant messages with reasoning_content."""
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_reasoning_to_chat_completion_message(
            {
                "type": "reasoning",
                "id": "rs_abc",
                "summary": [
                    {"type": "summary_text", "text": "Let me think about this..."}
                ],
            }
        )

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg.get("reasoning_content") == "Let me think about this..."
        assert msg.get("content") == ""

    def test_reasoning_with_multiple_summary_parts(self):
        """Reasoning with multiple summary parts should concatenate them."""
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_reasoning_to_chat_completion_message(
            {
                "type": "reasoning",
                "id": "rs_multi",
                "summary": [
                    {"type": "summary_text", "text": "First thought..."},
                    {"type": "summary_text", "text": "Second thought..."},
                ],
            }
        )

        assert len(result) == 1
        assert result[0]["reasoning_content"] == "First thought...Second thought..."

    def test_empty_reasoning_summary_returns_empty_list(self):
        """Reasoning with no summary text should return empty list."""
        result = LiteLLMCompletionResponsesConfig._transform_responses_api_reasoning_to_chat_completion_message(
            {"type": "reasoning", "id": "rs_empty", "summary": []}
        )
        assert result == []

        result = LiteLLMCompletionResponsesConfig._transform_responses_api_reasoning_to_chat_completion_message(
            {
                "type": "reasoning",
                "id": "rs_none",
                "summary": [{"type": "summary_text", "text": ""}],
            }
        )
        assert result == []

    def test_reasoning_input_item_in_full_input_transformation(self):
        """Reasoning items should be properly handled in full input transformation."""
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            [
                {"type": "message", "role": "user", "content": "What is 2+2?"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Calculating..."}],
                },
                {"type": "message", "role": "assistant", "content": "4"},
            ]
        )

        # Should have user message, reasoning message, and assistant message
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1].get("reasoning_content") == "Calculating..."
        assert messages[2]["role"] == "assistant"
        assert messages[2].get("content") == "4"

    def test_function_call_input_does_not_get_reasoning_content_by_default(self):
        """Assistant tool-call history should not add provider-specific fields by default."""
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_time",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "now",
                },
            ]
        )

        assistant_messages = [
            message for message in messages if message.get("role") == "assistant"
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].get("tool_calls")
        assert "reasoning_content" not in assistant_messages[0]

    def test_function_call_input_gets_reasoning_content_when_model_info_requires_it(
        self,
    ):
        """Deployment metadata can opt into reasoning_content compatibility."""
        request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
            model="mimo-v2.5-pro",
            input=[
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_time",
                    "arguments": "{}",
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "now",
                },
            ],
            responses_api_request={},
            custom_llm_provider="openai",
            model_info={"requires_tool_call_reasoning_content": True},
        )

        assistant_messages = [
            message
            for message in request["messages"]
            if message.get("role") == "assistant"
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0].get("reasoning_content") == " "
        assert request["_add_tool_call_reasoning_content"] is True

    def test_existing_tool_call_reasoning_content_is_preserved(self):
        """Existing reasoning_content on assistant tool calls should not be overwritten."""
        messages = LiteLLMCompletionResponsesConfig._ensure_assistant_tool_calls_have_reasoning_content(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "actual thinking",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"},
                        }
                    ],
                }
            ]
        )

        assert messages[0].get("reasoning_content") == "actual thinking"
