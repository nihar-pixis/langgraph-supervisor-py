from langgraph.pregel import PregelProtocol


class Agent(PregelProtocol):
    """Base class for an agent that can be used in a multi-agent workflow."""

    name: str
    """Agent name. Will be used as a node name in the multi-agent workflow."""
    is_entrypoint: bool
    """Whether the agent serves as an entrypoint in the multi-agent system"""
    always_handoff_to: list[str] | None
    """List of agent names (nodes) to always add edges to"""
