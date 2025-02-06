from langchain_core.language_models import LanguageModelLike

from langgraph.graph import StateGraph
from langgraph.prebuilt.chat_agent_executor import Prompt, StateSchemaType
from langgraph_multi_agent.workflow import create_multi_agent_workflow
from langgraph_multi_agent.agents import LangGraphAgent, ToolCallingAgent


def create_supervisor(
    agents: list[LangGraphAgent],
    *,
    supervisor_model: LanguageModelLike,
    supervisor_name: str = "supervisor",
    supervisor_prompt: Prompt | None = None,
    is_router: bool = False,
    schema: StateSchemaType | None = None,
) -> StateGraph:
    supervisor_can_handoff_to = [agent.name for agent in agents]
    workflow_agents = [
        ToolCallingAgent(
            name=supervisor_name,
            model=supervisor_model,
            tools=[],
            prompt=supervisor_prompt,
            can_handoff_to=supervisor_can_handoff_to,
            state_schema=schema,
            is_entrypoint=True,
        )
    ]

    for agent in agents:
        always_handoff_to = supervisor_name if not is_router else None
        agent_input_strategy = "full_history"
        agent_output_strategy = "last_message"

        updated_agent = agent.copy(
            always_handoff_to=always_handoff_to,
            agent_input_strategy=agent_input_strategy,
            agent_output_strategy=agent_output_strategy,
        )
        workflow_agents.append(updated_agent)

    return create_multi_agent_workflow(workflow_agents, schema=schema)
