from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    StateSchemaType,
    Prompt,
    create_react_agent,
)
from langgraph.pregel import PregelProtocol

from langgraph_multi_agent_supervisor.handoff import create_handoff_tool


@dataclass
class NamedAgent:
    name: str
    agent: PregelProtocol


def create_agent(
    *,
    name: str,
    model: LanguageModelLike,
    tools: list[Callable | BaseTool],
    prompt: Prompt | None,
    state_schema: StateSchemaType | None = None,
    can_handoff_to: list[str] | None = None,
    parallel_tool_calls: bool | None = False,
) -> NamedAgent:
    """Create a tool-calling agent.

    Args:
        name: Name of the agent
        model: Language model to use for the agent
        tools: Tools to use for the agent
        prompt: Prompt to use for the agent
        state_schema: State schema to use for the agent
        can_handoff_to: List of agent names (nodes) to hand off to (via tools)
        parallel_tool_calls: Whether the model is allowed to call tools in parallel.
            If None, defaults to the model's default behavior.
            Defaults to False (e.g., parallel tool calls are disabled)
    """
    # create handoff tools, if relevant
    handoff_tools = []
    if can_handoff_to is not None:
        handoff_tools = [
            create_handoff_tool(agent_name=agent_name) for agent_name in can_handoff_to
        ]

    all_tools = tools + handoff_tools
    # initialize ReAct agent
    if parallel_tool_calls is not None:
        model = model.bind_tools(all_tools, parallel_tool_calls=parallel_tool_calls)

    agent = create_react_agent(
        model=model,
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
    )
    return NamedAgent(name=name, agent=agent)
