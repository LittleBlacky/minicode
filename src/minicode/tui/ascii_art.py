"""ASCII Art and Animations for MiniCode TUI."""
import asyncio
import time
from typing import Iterator


class ASCIIArt:
    """ASCII art generator and animator."""

    # Cute cat ASCII art frames
    CAT_FRAMES = [
        r"""
  /\_/\
 ( o.o )
  > ^ <
        """,
        r"""
  /\_/\
  (o.o)
  (>^<)
        """,
        r"""
   /\ /\
  ( o.o )
   > ^ <
        """,
        r"""
  /\_/\
  ( -.- )
  (>^<)
        """,
    ]

    # Sleeping cat
    SLEEPING_CAT = r"""
    /\_/\
   ( =^.^= )
   (  ><   )
    V   V
        """

    # Typing cat
    TYPING_CAT = r"""
      /\_/\
     ( o.o )
     (> <)~
      /_\
   [typing]
        """

    # Happy cat
    HAPPY_CAT = r"""
   /\    /\
  ( ^.^ )/
   > ~ <
        """

    # Thinking cat
    THINKING_CAT = r"""
      /\_/\
     ( o.o )
      > ? <
        [thinking]
        """

    @classmethod
    def animate(cls, frames: list[str], delay: float = 0.5) -> Iterator[str]:
        """Animate through frames."""
        index = 0
        while True:
            yield frames[index % len(frames)]
            index += 1
            time.sleep(delay)

    @classmethod
    def get_cat_frame(cls, index: int) -> str:
        """Get a cat frame by index."""
        return cls.CAT_FRAMES[index % len(cls.CAT_FRAMES)]


class AnimatedLogo:
    """Animated logo display."""

    LOGO_FRAMES = [
        r"""
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
        """,
        r"""
   ██████╗ ██████╗ ███████╗ █████╗  ██████╗██╗  ██╗
  ██╔════╝██╔═══██╗██╔════╝██╔══██╗██╔════╝██║  ██║
  ██║     ██║   ██║█████╗  ███████║██║     ███████║
  ██║     ██║   ██║██╔══╝  ██╔══██║██║     ██╔══██║
  ╚██████╗╚██████╔╝███████╗██║  ██║╚██████╗██║  ██║
   ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        """,
    ]


class StatusAnimation:
    """Animated status indicators."""

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["—", "\\", "|", "/"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "star": ["✩", "✪", "✫", "✬", "✭", "✮", "✯", "✰"],
        "heart": ["♥", "❤", "💗", "💕", "💖", "💘", "💙", "💚"],
    }

    @classmethod
    def spinner(cls, style: str = "dots", delay: float = 0.1) -> Iterator[str]:
        """Generate spinner animation."""
        frames = cls.SPINNERS.get(style, cls.SPINNERS["dots"])
        index = 0
        while True:
            yield frames[index % len(frames)]
            index += 1
            time.sleep(delay)

    @classmethod
    def thinking(cls) -> Iterator[str]:
        """Thinking animation."""
        thoughts = ["Hmm...", "Let me think...", "Processing...", "Thinking...", "Analyzing..."]
        for i, thought in enumerate(thoughts):
            yield f"{thought} {cls.SPINNERS['dots'][i % len(cls.SPINNERS['dots'])]}"
            time.sleep(0.5)

    @classmethod
    def loading(cls) -> Iterator[str]:
        """Loading animation."""
        for i in range(100):
            bar = "█" * (i // 5) + "░" * (20 - i // 5)
            yield f"[{bar}] {i}%"
            time.sleep(0.02)


class CatAnimator:
    """Cat animation manager for TUI."""

    def __init__(self):
        self.current_frame = 0
        self.state = "idle"
        self.visible = True

    def set_state(self, state: str) -> None:
        """Set cat animation state."""
        self.state = state

    def get_art(self) -> str:
        """Get current ASCII art based on state."""
        if self.state == "sleeping":
            return ASCIIArt.SLEEPING_CAT
        elif self.state == "typing":
            return ASCIIArt.TYPING_CAT
        elif self.state == "happy":
            return ASCIIArt.HAPPY_CAT
        elif self.state == "thinking":
            return ASCIIArt.THINKING_CAT
        else:
            return ASCIIArt.get_cat_frame(self.current_frame)

    def next_frame(self) -> None:
        """Advance to next frame."""
        self.current_frame = (self.current_frame + 1) % len(ASCIIArt.CAT_FRAMES)

    def reset(self) -> None:
        """Reset animation."""
        self.current_frame = 0
        self.state = "idle"


async def animated_welcome(app) -> None:
    """Show animated welcome message."""
    from rich.console import Console
    from rich.live import Live

    console = Console()
    cat = CatAnimator()

    with Live(cat.get_art(), console=console, refresh_per_second=4) as live:
        for _ in range(20):
            cat.next_frame()
            live.update(cat.get_art())
            await asyncio.sleep(0.15)

        cat.set_state("happy")
        live.update(cat.get_art())
        await asyncio.sleep(0.5)


def print_ascii_cat() -> None:
    """Print ASCII cat art."""
    print(ASCIIArt.get_cat_frame(0))


def print_ascii_cat_animation(duration: float = 2.0) -> None:
    """Print animated ASCII cat art."""
    import sys
    import time

    start = time.time()
    index = 0
    while time.time() - start < duration:
        cat = ASCIIArt.get_cat_frame(index)
        # Move cursor up and print
        sys.stdout.write("\033[2J\033[H")  # Clear screen and home
        sys.stdout.write(cat)
        sys.stdout.flush()
        index = (index + 1) % len(ASCIIArt.CAT_FRAMES)
        time.sleep(0.3)

    # Final frame
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(ASCIIArt.CAT_FRAMES[0])
    sys.stdout.flush()