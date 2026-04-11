#!/usr/bin/env python3import os
import subprocess
from typing import Literal
import os

try:
    import readline

    # UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
    readline.parse_and_bind("set enable-meta-keybindings on")
except ImportError:
    pass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

load_dotenv(override=True)

MODEL_ID = os.environ["AGENCY_LLM_MODEL"]
BASE_URL = os.getenv("AGENCY_LLM_BASE_URL")
API_KEY = os.getenv("AGENCY_LLM_API_KEY")
PROVIDER = os.getenv("AGENCY_LLM_PROVIDER")
model = init_chat_model(
    MODEL_ID,
    model_provider=PROVIDER,  # explicitly set provider
    temperature=0,
    max_tokens=8000,
    base_url=BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)


@tool
def bash(command: str) -> str:
    """Run a shell command in the current workspace."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


tools = [bash]
tool_node = ToolNode(tools)

model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ----------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------
def call_model(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ----------------------------------------------------------------------
# Build and compile the graph
# ----------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ----------------------------------------------------------------------
# CLI loop (preserves original interactive style)
# ----------------------------------------------------------------------
def extract_text_from_message(msg) -> str:
    if isinstance(msg, AIMessage):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


if __name__ == "__main__":
    history = []  # list of LangChain messages (HumanMessage, AIMessage, ToolMessage)
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # Add user message to history
        history.append(HumanMessage(content=query))

        # Run the agent graph with the current history
        result = graph.invoke({"messages": history})

        # Update history with the final state (contains all intermediate messages)
        history = result["messages"]

        # Print the final assistant response (excluding tool call messages)
        final_message = history[-1]
        if isinstance(final_message, AIMessage) and not final_message.tool_calls:
            print(final_message.content)
        elif isinstance(final_message, AIMessage):
            # In case the last message has tool calls (shouldn't happen with __end__),
            # we still show the text part if any.
            if final_message.content:
                print(final_message.content)
        print()
