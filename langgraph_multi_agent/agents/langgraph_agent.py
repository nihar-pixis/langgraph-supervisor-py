from typing import Any, AsyncIterator, Iterator, Literal

from langchain_core.runnables.graph import Graph as DrawableGraph
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import PregelProtocol
from langgraph.pregel.types import StateSnapshot
from langgraph.types import StreamMode


AgentOutputStrategy = Literal["full_history", "last_message"]
"""How is the agent output added to the entire message history in the multi-agent workflow

- `full_history`: return the entire inner message history from the agent
- `last_message`: return the last message from the agent's inner history
"""


class LangGraphAgent(PregelProtocol):
    """Agent based on a LangGraph agent.

    Can be used with any agent that implements LangGraph PregelProtocol,
    including LangGraph's `StateGraph` (e.g., the prebuilt `create_react_agent`) and
    `RemoteGraph` (for interacting with agents deployed with LangGraph Platform)"""

    name: str
    """Agent name. Will be used as a node name in the multi-agent workflow."""
    agent: PregelProtocol
    """Underlying agent implementation."""
    is_entrypoint: bool
    """Whether the agent serves as an entrypoint in the multi-agent system"""
    always_handoff_to: str | None
    """List of agent names (nodes) to always add edges to"""
    agent_output_strategy: AgentOutputStrategy = "full_history"
    """How is the agent output added to the entire message history in the multi-agent workflow"""

    def __init__(
        self,
        *,
        name: str,
        agent: PregelProtocol,
        is_entrypoint: bool = False,
        always_handoff_to: str | None = None,
        agent_output_strategy: AgentOutputStrategy = "full_history",
    ):
        self.name = name
        self.agent = agent
        self.is_entrypoint = is_entrypoint
        self.always_handoff_to = always_handoff_to
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
        output = self.agent.invoke(input, config, **kwargs)
        return self._get_agent_output(output)

    async def ainvoke(
        self,
        input: dict[str, Any] | Any,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any] | Any:
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
