"""SubAgent module - Parent/Child agent pattern for task isolation."""
import asyncio
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage


class SubAgent:
    """Subagent with isolated context.

    Uses independent LangGraph instance with fresh message list.
    Returns only summary after completion.
    """

    def __init__(
        self,
        name: str,
        role: str,
        task: str,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-7",
    ):
        self.name = name
        self.role = role
        self.task = task
        self.model_provider = model_provider
        self.model_name = model_name
        self.result = ""

    async def run(self, additional_context: str = "") -> str:
        """Run the subagent task and return summary."""
        from minicode.agent.graph import create_agent_graph

        graph = create_agent_graph()

        system_prompt = f"""你是 {self.name}，角色是 {self.role}。
完成以下任务后，只返回任务结果的简要总结：

{task}

{additional_context}"""

        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=self.task),
        ]

        result = await graph.ainvoke({"messages": messages})

        # Extract last AI message as summary
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    self.result = msg.content
                    break

        return self.result

    def get_result(self) -> str:
        """Get the task result."""
        return self.result


class SubAgentPool:
    """Pool of subagents for parallel task execution."""

    def __init__(self, max_agents: int = 5):
        self.max_agents = max_agents
        self.agents: list[SubAgent] = []

    def create(
        self,
        name: str,
        role: str,
        task: str,
        model_provider: str = "anthropic",
        model_name: str = "claude-sonnet-4-7",
    ) -> SubAgent:
        """Create a new subagent."""
        if len(self.agents) >= self.max_agents:
            # Reuse oldest agent
            oldest = self.agents.pop(0)
        agent = SubAgent(name, role, task, model_provider, model_name)
        self.agents.append(agent)
        return agent

    async def run_all(
        self, tasks: list[tuple[str, str, str]], additional_context: str = ""
    ) -> list[str]:
        """Run multiple tasks in parallel."""
        agents = [
            self.create(name, role, task)
            for name, role, task in tasks
        ]

        results = await asyncio.gather(*[agent.run(additional_context) for agent in agents])
        return list(results)

    def clear(self) -> None:
        """Clear all agents."""
        self.agents.clear()


__all__ = ["SubAgent", "SubAgentPool"]
