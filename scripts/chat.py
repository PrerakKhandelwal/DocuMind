"""
scripts/chat.py
================
Interactive terminal chat with the Documentation Agent.

Usage:
    python scripts/chat.py
    python scripts/chat.py --version 2.3.1   # lock to a specific version

Special commands (type in the chat):
    /versions   — list all indexed versions
    /latest     — show the current latest version
    /help       — show available commands
    /quit       — exit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from src.agent import DocumentationAgent

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         Product Documentation Agent  v1.0                    ║
║       Type /help for commands  |  /quit to exit              ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
[bold]Available commands:[/bold]
  [cyan]/versions[/cyan]        — List all indexed document versions
  [cyan]/latest[/cyan]          — Show the current latest version
  [cyan]/version <ver>[/cyan]   — Lock queries to a specific version (e.g. /version 2.3.1)
  [cyan]/unlock[/cyan]          — Remove version lock (use latest)
  [cyan]/help[/cyan]            — Show this help message
  [cyan]/quit[/cyan]            — Exit the chat

[bold]Example questions:[/bold]
  • How do I configure the SSL certificate?
  • What are the default timeout settings in v2.3?
  • How do I enable clustering mode?
  • What changed in release 3.0 regarding authentication?
"""


@click.command()
@click.option("--version", "lock_version", default=None,
              help="Lock all queries to this version (e.g. 2.3.1)")
def main(lock_version: str):
    console.print(BANNER, style="bold blue")

    console.print("[bold yellow]Loading agent (this may take ~30s on first run)…[/bold yellow]")
    try:
        agent = DocumentationAgent()
    except Exception as e:
        console.print(f"[bold red]Failed to load agent: {e}[/bold red]")
        console.print("[yellow]Tip: Run 'python scripts/ingest.py' first to index your documents.[/yellow]")
        sys.exit(1)

    console.print("[bold green]Agent ready! Ask me anything about the product.[/bold green]\n")

    # Show indexed versions
    versions = agent._version_mgr.all_versions()
    latest = agent._version_mgr.get_latest()
    console.print(f"[dim]Indexed versions: {', '.join(versions) or 'none'}[/dim]")
    console.print(f"[dim]Latest version  : {latest or 'not set'}[/dim]\n")

    version_lock = lock_version  # may be None

    while True:
        try:
            # Build prompt suffix
            lock_label = f" [bold cyan][v{version_lock}][/bold cyan]" if version_lock else ""
            question = Prompt.ask(f"\n[bold green]You{lock_label}[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Goodbye![/bold]")
            break

        if not question.strip():
            continue

        # ── Handle commands ────────────────────────────────────────
        if question.startswith("/"):
            cmd = question.strip().lower()

            if cmd == "/quit" or cmd == "/exit":
                console.print("[bold]Goodbye![/bold]")
                break

            elif cmd == "/help":
                console.print(Panel(HELP_TEXT, title="Help", box=box.ROUNDED))

            elif cmd == "/versions":
                vlist = agent._version_mgr.all_versions()
                console.print(f"[bold]Indexed versions:[/bold] {', '.join(vlist) or 'none'}")

            elif cmd == "/latest":
                console.print(f"[bold]Latest version:[/bold] {agent._version_mgr.get_latest()}")

            elif cmd.startswith("/version "):
                requested = cmd.split(" ", 1)[1].strip()
                available = agent._version_mgr.all_versions()
                if requested in available:
                    version_lock = requested
                    console.print(f"[bold green]Version locked to: {version_lock}[/bold green]")
                else:
                    console.print(
                        f"[bold red]Version '{requested}' not indexed.[/bold red] "
                        f"Available: {', '.join(available)}"
                    )

            elif cmd == "/unlock":
                version_lock = None
                console.print("[bold green]Version lock removed. Using latest.[/bold green]")

            else:
                console.print(f"[yellow]Unknown command: {question}. Type /help.[/yellow]")
            continue

        # ── Inject version lock into question if set ───────────────
        effective_question = question
        if version_lock and "v" + version_lock not in question and version_lock not in question:
            effective_question = f"{question} (for version {version_lock})"

        # ── Ask the agent ──────────────────────────────────────────
        with console.status("[bold cyan]Searching documentation…[/bold cyan]"):
            try:
                result = agent.ask(effective_question)
            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")
                continue

        result.pretty_print()


if __name__ == "__main__":
    main()
