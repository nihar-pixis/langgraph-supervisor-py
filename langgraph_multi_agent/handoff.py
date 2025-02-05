from dataclasses import dataclass
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command


@dataclass(kw_only=True)
class HandoffConfig:
    """Configuration for the handoff."""

    agent_name: str
    """Name of the agent to hand off to."""
    create_task_description: bool = False
    """Whether to add the task description to the tool call."""


def create_handoff_tool(*, handoff_config: HandoffConfig):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{handoff_config.agent_name}"
    tool_message_content = f"Successfully transferred to {handoff_config.agent_name}"

    if handoff_config.create_task_description:

        @tool(tool_name)
        def handoff_to_agent(
            task_description: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            """Ask another agent for help.

            Args:
                task_description: detailed description of what the next agent should do, including all of the relevant context.
            """
            tool_message = ToolMessage(
                content=tool_message_content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=handoff_config.agent_name,
                graph=Command.PARENT,
                # NOTE: this message will be overwritten by the agent's output
                # when agent.agent_output_strategy == "tool_response"
                update={"messages": [tool_message]},
            )

    else:

        @tool(tool_name)
        def handoff_to_agent(
            tool_call_id: Annotated[str, InjectedToolCallId],
        ):
            """Ask another agent for help."""
            tool_message = ToolMessage(
                content=tool_message_content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=handoff_config.agent_name,
                graph=Command.PARENT,
                update={"messages": [tool_message]},
            )

    return handoff_to_agent
