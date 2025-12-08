import base64
import io
import json
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import chainlit as cl
from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ListToolsResult
from PIL import Image

load_dotenv(find_dotenv(), override=True)

# Constants
DEFAULT_IMAGE_RESPONSE = "Image generated successfully."
SYSTEM_MESSAGE_CONTENT = (
    "You are a helpful Yahoo Finance assistant. "
    "When tools return images, they will be automatically displayed to the user. "
    "Do NOT include image markdown syntax (![...]) in your responses. "
    "Simply describe the chart or image in text."
)
WELCOME_MESSAGE = (
    "Welcome to Yahoo Finance Chatbot! "
    "I can help you query stock information, news, and historical prices.\n\n"
    "Try asking me:\n"
    "- Get AAPL stock information\n"
    "- Recent TSLA news\n"
    "- Show NVDA price history for the past month"
)

# Determine which client to use based on environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
USE_LITELLM = LITELLM_API_KEY is not None and LITELLM_BASE_URL is not None

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1")


if USE_LITELLM:
    import litellm

    litellm.api_key = LITELLM_API_KEY
    litellm.api_base = LITELLM_BASE_URL
else:
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# Validate required environment variables
def validate_config() -> None:
    """Validate required environment variables are set."""
    if USE_LITELLM:
        if not LITELLM_API_KEY:
            raise ValueError("LITELLM_API_KEY must be set when using LiteLLM")
        if not LITELLM_BASE_URL:
            raise ValueError("LITELLM_BASE_URL must be set when using LiteLLM")
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI")


validate_config()


@asynccontextmanager
async def get_mcp_client() -> AsyncIterator[ClientSession]:
    """Create and manage MCP client connection to yfmcp server."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "yfmcp"],
    )

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def is_image_content(content: Any) -> bool:
    """Check if content is an image."""
    return hasattr(content, "data") and hasattr(content, "mimeType") and content.mimeType.startswith("image/")


async def chat_completion(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
    """Unified function to call chat completion API."""
    kwargs = {"model": DEFAULT_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if USE_LITELLM:
        return await litellm.acompletion(**kwargs)
    else:
        return await openai_client.chat.completions.create(**kwargs)


def extract_tool_result(result: Any, tool_name: str = "chart") -> tuple[str, list[cl.Image]]:
    """Extract text content and images from MCP tool result.

    Args:
        result: MCP tool result object
        tool_name: Name of the tool for image naming

    Returns:
        Tuple of (text_content, list_of_images)
    """
    tool_result = ""
    images = []

    if hasattr(result, "content") and result.content:
        logger.debug(f"Processing {len(result.content)} content items from {tool_name}")
        for idx, content in enumerate(result.content):
            # Handle text content
            if hasattr(content, "text"):
                tool_result += content.text
                logger.debug(f"Found text content: {len(content.text)} chars")
            # Handle image content
            elif is_image_content(content):
                image_data = base64.b64decode(content.data)
                # Use unique name for each image
                image_name = f"{tool_name}_{idx}" if len(result.content) > 1 else tool_name
                logger.debug(f"Found image: {image_name}, type: {content.mimeType}, size: {len(image_data)} bytes")

                # Convert WebP to PNG for better Chainlit compatibility
                if content.mimeType == "image/webp":
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        png_buffer = io.BytesIO()
                        img.save(png_buffer, format="PNG")
                        image_data = png_buffer.getvalue()
                        mime_type = "image/png"
                        logger.debug(f"Converted WebP to PNG, new size: {len(image_data)} bytes")
                    except Exception as e:
                        logger.warning(f"Failed to convert WebP to PNG: {e}")
                        mime_type = content.mimeType
                else:
                    mime_type = content.mimeType

                image = cl.Image(content=image_data, name=image_name, display="inline", mime=mime_type)
                images.append(image)

    return tool_result, images


def convert_mcp_tools_to_openai_format(tools_list: ListToolsResult) -> list[dict[str, Any]]:
    """Convert MCP tools to OpenAI tool format."""
    tools = []
    for tool in tools_list.tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        tools.append(tool_def)
    return tools


async def handle_error(error: Exception, context: str) -> None:
    """Handle and log errors with formatted message to user.

    Args:
        error: The exception that occurred
        context: Context description for the error (e.g., "initialization", "message handling")
    """
    logger.error(f"Error during {context}: {error}", exc_info=True)
    error_details = traceback.format_exc()
    error_message = f"Error during {context}: {error}\n\nDetails:\n```\n{error_details}\n```"
    await cl.Message(content=error_message).send()


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    try:
        # Initialize MCP client and keep connection alive
        mcp_context = get_mcp_client()
        session = await mcp_context.__aenter__()

        tools_list = await session.list_tools()
        tools = convert_mcp_tools_to_openai_format(tools_list)

        # Store MCP session and context manager for cleanup
        cl.user_session.set("mcp_session", session)
        cl.user_session.set("mcp_context", mcp_context)

        # Store tools and initialize message history with system message
        system_message = {
            "role": "system",
            "content": SYSTEM_MESSAGE_CONTENT,
        }
        cl.user_session.set("messages", [system_message])
        cl.user_session.set("tools", tools)

        await cl.Message(content=WELCOME_MESSAGE).send()

    except Exception as e:
        await handle_error(e, "initialization")
        raise


@cl.on_chat_end
async def end():
    """Clean up resources when chat ends."""
    try:
        mcp_context = cl.user_session.get("mcp_context")
        if mcp_context:
            await mcp_context.__aexit__(None, None, None)
            logger.info("MCP session closed successfully")
    except Exception as e:
        logger.error(f"Error closing MCP session: {e}", exc_info=True)


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and interact with MCP tools."""
    messages = cl.user_session.get("messages")
    tools = cl.user_session.get("tools")
    session = cl.user_session.get("mcp_session")

    if not session:
        await cl.Message(content="Error: MCP session not initialized. Please refresh the page.").send()
        return

    messages.append({"role": "user", "content": message.content})

    try:
        # Initial API call with tools
        response = await chat_completion(messages, tools)
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())

        # Process tool calls if any
        all_images = []
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Call MCP tool using reused session
                result = await session.call_tool(function_name, arguments=function_args)
                tool_result, images = extract_tool_result(result, function_name)

                # Collect images
                all_images.extend(images)

                # Add tool response to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_result if tool_result else DEFAULT_IMAGE_RESPONSE,
                    }
                )

            # Get final response after tool calls
            final_response = await chat_completion(messages)
            final_message = final_response.choices[0].message.content
            messages.append({"role": "assistant", "content": final_message})
        else:
            final_message = assistant_message.content
            if not final_message:
                final_message = "Sorry, I couldn't understand your question."

        # Send response to user with images
        await cl.Message(content=final_message, elements=all_images if all_images else None).send()

    except Exception as e:
        await handle_error(e, "message handling")
