from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.chat_agent_executor import StateSchemaType

from langgraph_multi_agent.agents import LangGraphAgent, ToolCallingAgent


def _validate_agent_handoffs(agent_name_to_agent: dict[str, LangGraphAgent]) -> None:
    for agent_name, agent in agent_name_to_agent.items():
        if not isinstance(agent, ToolCallingAgent) or not agent.can_handoff_to:
            continue

        for agent_name in agent.can_handoff_to:
            target_agent = agent_name_to_agent.get(agent_name)
            if not target_agent:
                raise ValueError(
                    f"Agent '{agent_name}' has a handoff config for non-existent agent '{agent_name}'"
                )


def create_multi_agent_workflow(
    agents: list[LangGraphAgent],
    schema: StateSchemaType | None = None,
) -> StateGraph:
    agent_name_to_agent = {agent.name: agent for agent in agents}
    _validate_agent_handoffs(agent_name_to_agent)

    builder = StateGraph(schema or MessagesState)
    for agent in agents:
        builder.add_node(agent.name, agent)

        if agent.is_entrypoint:
            builder.add_edge(START, agent.name)

        # useful for supervisor-style architectures
        if agent.always_handoff_to:
            builder.add_edge(agent.name, agent.always_handoff_to)

    return builder
