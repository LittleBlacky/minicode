# MiniCode

A terminal coding agent powered by LangGraph.

## Features

- **Graph-based Agent**: Workflow orchestration using LangGraph
- **13+ Tools**: Bash, file operations, search, tasks, teammates
- **Checkpoint**: Session persistence and state recovery
- **Rich Interfaces**: REPL interface and Textual TUI
- **Multi-Model**: Supports Anthropic Claude and OpenAI GPT

## Installation

```bash
pip install -e .
```

## Usage

### Single Command

```bash
minicode "帮我创建一个 hello.py 文件"
```

### Interactive Mode

```bash
minicode
```

Launches REPL mode with interactive input.

## Available Tools

| Tool           | Description             |
| -------------- | ----------------------- |
| read_file      | Read file contents      |
| write_file     | Write file contents     |
| edit_file      | Edit existing files     |
| bash_run       | Execute Bash commands   |
| glob_tool      | Find files by pattern   |
| grep_tool      | Search file contents    |
| TaskCreate     | Create tasks            |
| TodoWrite      | Write Todo list         |
| spawn_teammate | Spawn AI teammates      |
| background_run | Run tasks in background |
| cron_create    | Create scheduled tasks  |
| memory_save    | Save persistent memory  |
| skill_create   | Create skills           |

## Configuration

Create `~/.mini-agent-cli/config.json`:

```json
{
  "model": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-7"
  },
  "permissions": {
    "mode": "default"
  },
  "features": {
    "auto_compact": true,
    "team_enabled": false,
    "skills_enabled": true
  }
}
```

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
```

## License

MIT

