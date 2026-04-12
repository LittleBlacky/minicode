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
from langgraph.checkpoint.memory import MemorySaver
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
    "Use bash to inspect and change the workspace. Act first, then report clearly.\n"
    "LangGraph native: Checkpoint persistence for session recovery."
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

# Compile with checkpoint for session persistence (LangGraph native)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_session_config(thread_id: str) -> dict:
    """LangGraph native: Get session config for checkpointing."""
    return {"configurable": {"thread_id": thread_id}}


# ----------------------------------------------------------------------
# CLI loop with checkpoint persistence
# ----------------------------------------------------------------------
def extract_text_from_message(msg) -> str:
    if isinstance(msg, AIMessage):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


if __name__ == "__main__":
    thread_id = "agent_session_1"
    config = get_session_config(thread_id)

    # Resume from checkpoint if exists
    existing = graph.get_state(config)
    if existing and existing.values.get("messages"):
        print(
            f"[Resuming session {thread_id} with {len(existing.values['messages'])} messages]\n"
        )

    print("Agent Loop (phase1) - LangGraph Native Patterns")
    print("Features: Checkpoint persistence, streaming output")
    print("Type 'exit' or 'q' to quit\n")

    while True:
        try:
            query = input(f"\033[36m{thread_id} >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # Get existing messages from checkpoint
        existing = graph.get_state(config)
        existing_msgs = (
            existing.values.get("messages", []) if existing and existing.values else []
        )
        existing_msgs.append(HumanMessage(content=query))

        # Run the agent graph with checkpoint
        result = graph.invoke({"messages": existing_msgs}, config)

        # Get updated messages from checkpoint
        updated = graph.get_state(config)
        history = (
            updated.values.get("messages", [])
            if updated and updated.values
            else result["messages"]
        )

        # Print the final assistant response
        final_message = history[-1]
        if isinstance(final_message, AIMessage) and not final_message.tool_calls:
            print(final_message.content)
        elif isinstance(final_message, AIMessage):
            if final_message.content:
                print(final_message.content)
        print()
