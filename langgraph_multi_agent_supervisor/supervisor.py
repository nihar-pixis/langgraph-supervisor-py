from dataclasses import dataclass
from typing import Callable, Literal

from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.pregel import PregelProtocol
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.chat_agent_executor import StateSchemaType, Prompt

from langgraph_multi_agent_supervisor.agent import NamedAgent, create_agent


AgentOutputStrategy = Literal["full_history", "last_message"]
"""How is the agent output added to the entire message history in the multi-agent workflow

- `full_history`: return the entire inner message history from the agent
- `last_message`: return the last message from the agent's inner history
"""


def _make_call_agent(
    agent: PregelProtocol, agent_output_strategy: AgentOutputStrategy
) -> Callable:
    if agent_output_strategy not in AgentOutputStrategy.__args__:
        raise ValueError(
            f"Invalid agent output strategy: {agent_output_strategy}. "
            f"Needs to be one of {AgentOutputStrategy.__args__}"
        )

    def call_agent(state):
        output = agent.invoke(state)
        if agent_output_strategy == "full_history":
            return output
        elif agent_output_strategy == "last_message":
            return {"messages": output["messages"][-1]}

    return call_agent


def create_supervisor(
    agents: list[NamedAgent],
    *,
    model: LanguageModelLike,
    tools: list[Callable | BaseTool] | None = None,
    prompt: Prompt | None = None,
    state_schema: StateSchemaType | None = None,
    is_router: bool = False,
    agent_output_strategy: AgentOutputStrategy = "last_message",
    supervisor_name: str = "supervisor",
) -> StateGraph:
    """Create a multi-agent supervisor.

    Args:
        agents: List of agents to supervise
        model: Language model to use for the supervisor
        tools: Tools to use for the supervisor
        prompt: Prompt to use for the supervisor
        state_schema: State schema to use for the supervisor graph
        is_router: Whether the supervisor is a router (i.e. agents can respond to the user directly),
            or agents always return control back to the supervisor
        agent_output_strategy: How the agent output is added to the entire message history in the multi-agent workflow
        supervisor_name: Name of the supervisor node
    """
    supervisor_agent = create_agent(
        name=supervisor_name,
        model=model,
        tools=tools or [],
        prompt=prompt,
        can_handoff_to=[agent.name for agent in agents],
        state_schema=state_schema,
    )

    builder = StateGraph(state_schema or MessagesState)
    builder.add_node(supervisor_agent.name, supervisor_agent.agent)
    builder.add_edge(START, supervisor_agent.name)
    for agent in agents:
        builder.add_node(
            agent.name, _make_call_agent(agent.agent, agent_output_strategy)
        )
        if not is_router:
            builder.add_edge(agent.name, supervisor_agent.name)

    return builder
