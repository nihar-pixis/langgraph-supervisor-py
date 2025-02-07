from typing import Callable, Literal

from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    StateSchemaType,
    Prompt,
    create_react_agent,
)

from langgraph_multi_agent_supervisor.handoff import (
    create_handoff_tool,
    create_handoff_back_messages,
)


OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""


def _make_call_agent(
    agent: CompiledStateGraph,
    agent_output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
) -> Callable:
    if agent_output_mode not in OutputMode.__args__:
        raise ValueError(
            f"Invalid agent output mode: {agent_output_mode}. "
            f"Needs to be one of {OutputMode.__args__}"
        )

    def call_agent(state: MessagesState) -> MessagesState:
        output = agent.invoke(state)
        messages = output["messages"]
        if agent_output_mode == "full_history":
            pass
        elif agent_output_mode == "last_message":
            messages = messages[-1:]
        else:
            raise ValueError(
                f"Invalid agent output mode: {agent_output_mode}. "
                f"Needs to be one of {OutputMode.__args__}"
            )

        if add_handoff_back_messages:
            messages.extend(create_handoff_back_messages(agent.name, supervisor_name))

        return {"messages": messages}

    return call_agent


def create_supervisor(
    agents: list[CompiledStateGraph],
    *,
    model: LanguageModelLike,
    tools: list[Callable | BaseTool] | None = None,
    prompt: Prompt | None = None,
    state_schema: StateSchemaType | None = None,
    is_router: bool = False,
    agent_output_mode: OutputMode = "last_message",
    add_handoff_back_messages: bool = True,
    supervisor_name: str = "supervisor",
) -> StateGraph:
    """Create a multi-agent supervisor.

    Args:
        agents: List of agents to manage
        model: Language model to use for the supervisor
        tools: Tools to use for the supervisor
        prompt: Optional prompt to use for the supervisor. Can be one of:
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        state_schema: State schema to use for the supervisor graph. Defaults to MessagesState
        is_router: Whether the supervisor is a router (i.e. agents can respond to the user directly),
            or agents always return control back to the supervisor. Defaults to False.
        agent_output_mode: Mode for adding agent outputs to the message history in the multi-agent workflow.
            Can be one of:
            - `full_history`: add the entire agent message history
            - `last_message`: add only the last message
            Defaults to `last_message`
        add_handoff_back_messages: Whether to add a pair of (AIMessage, ToolMessage) to the message history
            when returning control to the supervisor to indicate that a handoff has occurred.
            Defaults to True
        supervisor_name: Name of the supervisor node. Defaults to "supervisor"
    """
    agent_names = set()
    for agent in agents:
        if agent.name is None or agent.name == "LangGraph":
            raise ValueError(
                "Please specify a name when you create your agent, either via `create_react_agent(..., name=agent_name)` "
                "or via `graph.compile(name=name)`."
            )

        if agent.name in agent_names:
            raise ValueError(
                f"Agent with name '{agent.name}' already exists. Agent names must be unique."
            )

        agent_names.add(agent.name)

    handoff_tools = [create_handoff_tool(agent_name=agent.name) for agent in agents]
    all_tools = (tools or []) + handoff_tools
    supervisor_agent = create_react_agent(
        name=supervisor_name,
        model=model.bind_tools(all_tools, parallel_tool_calls=False),
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
    )

    builder = StateGraph(state_schema or MessagesState)
    builder.add_node(supervisor_agent)
    builder.add_edge(START, supervisor_agent.name)
    for agent in agents:
        builder.add_node(
            agent.name,
            _make_call_agent(
                agent,
                agent_output_mode,
                add_handoff_back_messages,
                supervisor_name,
            ),
        )
        if not is_router:
            builder.add_edge(agent.name, supervisor_agent.name)

    return builder
