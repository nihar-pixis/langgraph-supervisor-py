from typing import Callable
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.prebuilt.chat_agent_executor import (
    StateSchemaType,
    Prompt,
    create_react_agent,
)

from langgraph_multi_agent.agents.langgraph_agent import (
    LangGraphAgent,
    AgentInputStrategy,
    AgentOutputStrategy,
)
from langgraph_multi_agent.handoff import HandoffConfig, create_handoff_tool


class ToolCallingAgent(LangGraphAgent):
    """Tool-calling agent."""

    model: LanguageModelLike
    """Language model to use for the agent"""
    tools: list[Callable | BaseTool]
    """Tools to use for the agent"""
    prompt: Prompt | None
    """Prompt to use for the agent"""
    state_schema: StateSchemaType
    """State schema to use for the agent"""
    can_handoff_to: list[str | HandoffConfig] | None
    """List of agent names (nodes) to hand off to (via tools) or HandoffConfig objects"""
    parallel_tool_calls: bool | None
    """Whether the model is allowed to call tools in parallel.
        If None, defaults to the model's default behavior.
        Defaults to False (e.g., parallel tool calls are disabled)
    """

    def __init__(
        self,
        *,
        name: str,
        model: LanguageModelLike,
        tools: list[Callable | BaseTool],
        prompt: Prompt | None = None,
        state_schema: StateSchemaType = None,
        is_entrypoint: bool = False,
        can_handoff_to: list[str | HandoffConfig] | None = None,
        always_handoff_to: list[str] | None = None,
        agent_input_strategy: AgentInputStrategy = "full_history",
        agent_output_strategy: AgentOutputStrategy = "full_history",
        parallel_tool_calls: bool | None = False,
    ):
        self.model = model
        self.tools = tools
        self.prompt = prompt
        self.state_schema = state_schema
        self.parallel_tool_calls = parallel_tool_calls
        self.can_handoff_to = can_handoff_to

        # create handoff tools, if relevant
        handoff_tools = []
        if self.can_handoff_to is not None:
            self.handoff_configs = [
                HandoffConfig(agent_name=handoff_to)
                if isinstance(handoff_to, str)
                else handoff_to
                for handoff_to in self.can_handoff_to
            ]
            handoff_tools = [
                create_handoff_tool(handoff_config=handoff_config)
                for handoff_config in self.handoff_configs
            ]

        all_tools = self.tools + handoff_tools
        # initialize ReAct agent
        if self.parallel_tool_calls is not None:
            self.model = self.model.bind_tools(
                all_tools, parallel_tool_calls=self.parallel_tool_calls
            )

        agent = create_react_agent(
            model=self.model,
            tools=all_tools,
            prompt=self.prompt,
            state_schema=self.state_schema,
        )
        super().__init__(
            name=name,
            agent=agent,
            is_entrypoint=is_entrypoint,
            always_handoff_to=always_handoff_to,
            agent_input_strategy=agent_input_strategy,
            agent_output_strategy=agent_output_strategy,
        )
        self._validate()

    def _validate(self):
        if (
            self.agent_output_strategy == "tool_response"
            and self.agent_input_strategy != "tool_call"
        ):
            raise ValueError(
                "`agent_output_strategy='tool_response'` must be used with `agent_input_strategy='tool_call'`, got "
                f"`agent_input_strategy='{self.agent_input_strategy}'` and `agent_output_strategy='{self.agent_output_strategy}'`"
            )

        if self.agent_output_strategy == "last_message" and self.can_handoff_to:
            raise ValueError(
                "Cannot use `agent_output_strategy='last_message'` when handoffs are enabled. "
                "Please use `agent_output_strategy='full_history'` or `agent_output_strategy='tool_response'` instead."
            )
