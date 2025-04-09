import sys


def pydantic_mistral_gcp_patch():
    from mistralai_gcp import (
        CompletionChunk as MistralCompletionChunk,
    )
    from mistralai_gcp import (
        Content as MistralContent,
    )
    from mistralai_gcp import (
        ContentChunk as MistralContentChunk,
    )
    from mistralai_gcp import (
        FunctionCall as MistralFunctionCall,
    )
    from mistralai_gcp import (
        OptionalNullable as MistralOptionalNullable,
    )
    from mistralai_gcp import (
        TextChunk as MistralTextChunk,
    )
    from mistralai_gcp import (
        ToolChoiceEnum as MistralToolChoiceEnum,
    )
    from mistralai_gcp.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
    )
    from mistralai_gcp.models import (
        CompletionEvent as MistralCompletionEvent,
    )
    from mistralai_gcp.models import (
        Messages as MistralMessages,
    )
    from mistralai_gcp.models import (
        Tool as MistralTool,
    )
    from mistralai_gcp.models import (
        ToolCall as MistralToolCall,
    )
    from mistralai_gcp.models.assistantmessage import AssistantMessage as MistralAssistantMessage
    from mistralai_gcp.models.function import Function as MistralFunction
    from mistralai_gcp.models.systemmessage import SystemMessage as MistralSystemMessage
    from mistralai_gcp.models.toolmessage import ToolMessage as MistralToolMessage
    from mistralai_gcp.models.usermessage import UserMessage as MistralUserMessage
    from mistralai_gcp.types.basemodel import Unset as MistralUnset
    from mistralai_gcp.utils.eventstreaming import EventStreamAsync as MistralEventStreamAsync

    # from mistralai_gcp.models.imageurl import (
    #     ImageURL as MistralImageURL,
    #     ImageURLChunk as MistralImageURLChunk,
    # )

    sys.modules["pydantic_ai.models.mistral"].MistralUserMessage = MistralUserMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralSystemMessage = MistralSystemMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralAssistantMessage = MistralAssistantMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunction = MistralFunction  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolMessage = MistralToolMessage  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralChatCompletionResponse = MistralChatCompletionResponse  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionEvent = MistralCompletionEvent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralMessages = MistralMessages  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTool = MistralTool  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolCall = MistralToolCall  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralUnset = MistralUnset  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralEventStreamAsync = MistralEventStreamAsync  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralOptionalNullable = MistralOptionalNullable  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralTextChunk = MistralTextChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralToolChoiceEnum = MistralToolChoiceEnum  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralCompletionChunk = MistralCompletionChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContent = MistralContent  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralContentChunk = MistralContentChunk  # type: ignore
    sys.modules["pydantic_ai.models.mistral"].MistralFunctionCall = MistralFunctionCall  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURL = MistralImageURL  # type: ignore
    # sys.modules['pydantic_ai.models.mistral'].MistralImageURLChunk = MistralImageURLChunk  # type: ignore
