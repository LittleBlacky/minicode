"""CLI Entry Point"""
import sys
import asyncio
from pathlib import Path
import argparse

from minicode.agent.runner import AgentRunner
from minicode.agent.runner import run_interactive
from minicode.agent.session import SessionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiniCode - 智能终端编码助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  minicode                          Start with TUI interface
  minicode "fix the bug"            Run a single task with TUI
  minicode --model claude-sonnet-4-7   Use different model
  minicode --workdir /path/to/proj  Set working directory

Tips:
  @filename  - Reference a file in your message
  /          - Show all commands
  /help      - Show help
        """,
    )
    parser.add_argument("task", nargs="?", help="Task to execute")
    parser.add_argument("--model", "-m", default=None, help="Model name")
    parser.add_argument("--provider", "-p", default=None, help="Model provider")
    parser.add_argument("--workdir", "-w", type=Path, help="Working directory")
    parser.add_argument("--session", "-s", default="default", help="Session ID")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint")
    parser.add_argument("--db", help="SQLite DB path for checkpointing")
    return parser.parse_args()


async def run_task(runner: AgentRunner, task: str) -> None:
    """Run a single task."""
    from langchain_core.messages import HumanMessage

    print(f"\n[会话: {runner.thread_id}] {task}\n")

    messages = [HumanMessage(content=task)]
    result = await runner.run(messages)

    # Print last assistant message
    msgs = result.get("messages", [])
    for msg in reversed(msgs):
        if hasattr(msg, "content") and msg.content:
            print(f"\n{msg.content}")
            break


def main():
    import os
    args = parse_args()

    # 只有明确传入参数时才设置环境变量，否则让 create_chat_model 从 config.json 读取
    if args.provider is not None:
        os.environ["MINICODE_PROVIDER"] = args.provider
    if args.model is not None:
        os.environ["MINICODE_MODEL"] = args.model

    # Always start with TUI mode
    print("Starting MiniCode TUI...")
    from minicode.tui.app import run_tui
    runner = AgentRunner(
        use_checkpoint=not args.no_checkpoint,
        workdir=args.workdir,
        thread_id=args.session,
    )

    # If a task is provided, execute it then show TUI
    if args.task:
        asyncio.run(run_task(runner, args.task))

    # Always show TUI
    asyncio.run(run_tui(runner))


if __name__ == "__main__":
    main()
