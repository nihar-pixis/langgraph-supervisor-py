from typing import Any, AsyncIterator, Iterator, Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables.graph import Graph as DrawableGraph
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import PregelProtocol
from langgraph.pregel.types import StateSnapshot
from langgraph.types import StreamMode

from langgraph_multi_agent.agents.agent import Agent


AgentInputStrategy = Literal["full_history", "tool_call"]
"""How is information passed to the agent

- `full_history`: pass the entire message history of the multi-agent workflow
    as an input to the current agent
- `tool_call`: pass `task_description` populated via a tool call by the previous agent
    during the handoff
"""

AgentOutputStrategy = Literal["full_history", "last_message", "tool_response"]
"""How is the agent output added to the entire message history in the multi-agent workflow

- `full_history`: return the entire inner message history from the agent
- `last_message`: return the last message from the agent's inner history
- `tool_response`: return the last message from the agent's inner history as a ToolMessage
    should be used only when the agent is invoked as a tool call.
"""


def _get_agent_input_from_tool_call(
    input: MessagesState, agent_name: str
) -> MessagesState:
    # by this point the multi-agent workflow state
    # will contain an AI message w/ handoff tool call AND a tool message
    # returned by the handoff tool
    if len(input["messages"]) < 2:
        raise ValueError(
            f"To invoke the agent as a tool, the input must have at least two messages (AI message w/ tool calls and a corresponding tool message), got {input['messages']}"
        )

    last_ai_message = input["messages"][-2]
    last_tool_message = input["messages"][-1]
    if last_ai_message.type != "ai" or last_tool_message.type != "tool":
        raise ValueError(
            f"To invoke the agent as a tool, the last two messages in the input must be AI message w/ tool calls and a corresponding tool message, got {input['messages'][-2:]}"
        )

    if len(last_ai_message.tool_calls) != 1:
        raise ValueError(
            f"To invoke the agent as a tool, the last AI message must have exactly one tool call, got {len(last_ai_message.tool_calls)}"
        )

    tool_call = last_ai_message.tool_calls[-1]
    if agent_name not in tool_call["name"]:
        raise ValueError(
            f"To invoke the agent as a tool, the last AI message must have a tool call with the name {agent_name}, got {last_ai_message.tool_calls[-1]['name']}"
        )

    if "task_description" not in tool_call["args"]:
        raise ValueError(
            f"To invoke the agent as a tool, the last AI message must have a tool call with task_description in the args, got the following args: {last_ai_message.tool_calls[-1]['args']}"
        )

    task_description = tool_call["args"]["task_description"]
    return {
        "messages": [
            HumanMessage(
                content=task_description,
                additional_kwargs={
                    "tool_call_id": tool_call["id"],
                    "tool_call_name": tool_call["name"],
                    "tool_message_id": last_tool_message.id,
                },
            )
        ]
    }


def _get_agent_output_as_tool_response(output: MessagesState):
    tool_call_info = output["messages"][0].additional_kwargs
    tool_msg = ToolMessage(
        content=output["messages"][-1].content,
        name=tool_call_info["tool_call_name"],
        tool_call_id=tool_call_info["tool_call_id"],
        # NOTE: we're reusing the same message ID to replace the tool message
        # returned by the handoff tool. This assumes the state is based on MessagesState
        # or is using a `messages` key with `add_messages` reducer
        id=tool_call_info["tool_message_id"],
    )
    return {"messages": [tool_msg]}


class LangGraphAgent(Agent):
    """Agent based on a LangGraph agent.

    Can be used with any agent that implements LangGraph PregelProtocol,
    including LangGraph's `StateGraph` (e.g., the prebuilt `create_react_agent`) and
    `RemoteGraph` (for interacting with agents deployed with LangGraph Platform)"""

    agent: PregelProtocol
    """Underlying agent implementation."""
    agent_input_strategy: AgentInputStrategy = "full_history"
    """How is information passed to the agent"""
    agent_output_strategy: AgentOutputStrategy = "full_history"
    """How is the agent output added to the entire message history in the multi-agent workflow"""

    def __init__(
        self,
        *,
        name: str,
        agent: PregelProtocol,
        is_entrypoint: bool = False,
        always_handoff_to: list[str] | None = None,
        agent_input_strategy: AgentInputStrategy = "full_history",
        agent_output_strategy: AgentOutputStrategy = "full_history",
    ):
        self.name = name
        self.agent = agent
        self.is_entrypoint = is_entrypoint
        self.always_handoff_to = always_handoff_to
        self.agent_input_strategy = agent_input_strategy
        self.agent_output_strategy = agent_output_strategy

    def _get_agent_input(self, input: dict[str, Any]) -> dict[str, Any]:
        if self.agent_input_strategy == "full_history":
            return input

        elif self.agent_input_strategy == "tool_call":
            return _get_agent_input_from_tool_call(input, self.name)
        else:
            raise ValueError(
                f"Invalid agent input strategy: {self.agent_input_strategy}. "
                f"Needs to be one of {AgentInputStrategy.__args__}"
            )

    def _get_agent_output(self, output: dict[str, Any]) -> dict[str, Any]:
        if self.agent_output_strategy == "full_history":
            return output
        elif self.agent_output_strategy == "last_message":
            return {"messages": output["messages"][-1]}
        elif self.agent_output_strategy == "tool_response":
            return _get_agent_output_as_tool_response(output)
        else:
            raise ValueError(
                f"Invalid agent output strategy: {self.agent_output_strategy}. "
                f"Needs to be one of {AgentOutputStrategy.__args__}"
            )

    def copy(self, **update: dict[str, Any]) -> "LangGraphAgent":
        attrs = {**self.__dict__, **update}
        return self.__class__(**attrs)

    def invoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any] | Any:
        input = self._get_agent_input(input)
        output = self.agent.invoke(input, config, **kwargs)
        return self._get_agent_output(output)

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any] | Any:
        input = self._get_agent_input(input)
        output = await self.agent.ainvoke(input, config, **kwargs)
        return self._get_agent_output(output)

    # Below methods are unchanged and behave the same as the underlying agent

    def stream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        **kwargs,
    ) -> Iterator[dict[str, Any] | Any]:
        return self.agent.stream(input, config, stream_mode=stream_mode, **kwargs)

    async def astream(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | list[StreamMode] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any] | Any]:
        async for chunk in self.agent.astream(
            input, config, stream_mode=stream_mode, **kwargs
        ):
            yield chunk

    def with_config(
        self, config: RunnableConfig | None = None, **kwargs: Any
    ) -> CompiledStateGraph:
        return self.agent.with_config(config, **kwargs)

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        return self.agent.update_state(config, values, as_node)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig:
        return await self.agent.aupdate_state(config, values, as_node)

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        return self.agent.get_state(config, subgraphs=subgraphs)

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        return await self.agent.aget_state(config, subgraphs=subgraphs)

    def get_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> DrawableGraph:
        return self.agent.get_graph(config, xray=xray)

    async def aget_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> DrawableGraph:
        return await self.agent.aget_graph(config, xray=xray)

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        return self.agent.get_state_history(
            config, filter=filter, before=before, limit=limit
        )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        async for state in self.agent.aget_state_history(
            config, filter=filter, before=before, limit=limit
        ):
            yield state
