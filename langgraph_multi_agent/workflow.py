from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.chat_agent_executor import StateSchemaType

from langgraph_multi_agent.agents import LangGraphAgent, ToolCallingAgent
from langgraph_multi_agent.handoff import HandoffConfig


def _validate_agent_handoffs(agent_name_to_agent: dict[str, LangGraphAgent]) -> None:
    for agent_name, agent in agent_name_to_agent.items():
        if not isinstance(agent, ToolCallingAgent) or not agent.can_handoff_to:
            continue

        for handoff_config in agent.can_handoff_to:
            if isinstance(handoff_config, str):
                handoff_config = HandoffConfig(agent_name=handoff_config)

            target_agent = agent_name_to_agent.get(handoff_config.agent_name)
            if not target_agent:
                raise ValueError(
                    f"Agent '{agent_name}' has a handoff config for non-existent agent '{handoff_config.agent_name}'"
                )

            if (
                handoff_config.create_task_description
                and target_agent.agent_input_strategy != "tool_call"
            ):
                raise ValueError(
                    f"Agent '{agent_name}' has a handoff config that requests a task description ({handoff_config}) "
                    f"but the target agent is configured with input strategy '{target_agent.agent_input_strategy}'. "
                    "To use a handoff with a task description, the target agent must be configured with input strategy 'tool_call'."
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
