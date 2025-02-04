from langgraph.graph import StateGraph, START
from langgraph.prebuilt.chat_agent_executor import StateSchemaType, AgentState

from langgraph_multi_agent.agents import Agent


def create_multi_agent_workflow(
    agents: list[Agent],
    schema: StateSchemaType | None = None,
) -> StateGraph:
    builder = StateGraph(schema or AgentState)
    for agent in agents:
        builder.add_node(agent.name, agent)

        if agent.is_entrypoint:
            builder.add_edge(START, agent.name)

        # useful for supervisor-style architectures
        for target in agent.always_handoff_to or []:
            builder.add_edge(agent.name, target)

    return builder
