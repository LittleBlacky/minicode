"""Main entry point for minicode CLI."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from minicode.agent.runner import AgentRunner
from minicode.repl.repl import start_repl
from minicode.services.config import get_config_manager


def get_workdir(args: argparse.Namespace) -> Path:
    """Determine working directory."""
    if args.workdir:
        return Path(args.workdir).resolve()
    return Path.cwd()


async def run_single(runner: AgentRunner, prompt: str, workdir: Path) -> None:
    """Run a single prompt."""
    from langchain_core.messages import HumanMessage

    os.chdir(workdir)
    messages = [HumanMessage(content=prompt)]
    result = await runner.run(messages)
    print(result)


async def run_repl(runner: AgentRunner, workdir: Path) -> None:
    """Run interactive REPL."""
    os.chdir(workdir)
    await start_repl(runner)


async def run_tui(runner: AgentRunner, workdir: Path) -> None:
    """Run Textual TUI."""
    from minicode.tui.app import run_tui as start_tui
    os.chdir(workdir)
    await start_tui(runner)


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCode - Claude-style coding agent")
    parser.add_argument("prompt", nargs="*", help="Single prompt to execute")
    parser.add_argument("-w", "--workdir", help="Working directory")
    parser.add_argument("--model", default="claude-sonnet-4-7", help="Model name")
    parser.add_argument("--provider", default="anthropic", help="Model provider")
    parser.add_argument("--checkpoint", action="store_true", help="Enable checkpointing")
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    parser.add_argument("--tui", action="store_true", help="Use Textual TUI interface")
    parser.add_argument("--plain", action="store_true", help="Use plain REPL (no TUI)")

    args = parser.parse_args()

    if args.version:
        print("MiniCode 0.1.0")
        sys.exit(0)

    workdir = get_workdir(args)

    runner = AgentRunner(
        model_provider=args.provider,
        model_name=args.model,
        use_checkpoint=args.checkpoint,
    )

    if args.prompt:
        prompt = " ".join(args.prompt)
        asyncio.run(run_single(runner, prompt, workdir))
    elif args.plain:
        asyncio.run(run_repl(runner, workdir))
    else:
        asyncio.run(run_tui(runner, workdir))


if __name__ == "__main__":
    main()