"""Rich-based rendering: track tables and progress bars, ASCII-safe."""

from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import Progress, TextColumn
from rich.table import Table


def track_table(tracks: Sequence[Any], console: Optional[Console] = None) -> None:
    """
    Render a summary table of tracks to the console.

    Parameters
    ----------
    tracks : sequence
        Track objects with ``id``, ``status``, ``state`` (and optionally
        ``covariance``) attributes, e.g. from ``MultiTargetTracker.process``.
    console : rich.console.Console, optional
        Target console; defaults to stderr. Output is ASCII-only
        (``box.ASCII``) to satisfy the console-encoding contract.
    """
    console = console or Console(stderr=True)
    table = Table(title="Tracks", box=box.ASCII, safe_box=True)
    table.add_column("id", justify="right")
    table.add_column("status")
    table.add_column("position")
    table.add_column("speed", justify="right")
    for t in tracks:
        state = np.asarray(t.state, dtype=float).ravel()
        # Convention: interleaved [x, vx, y, vy, ...]; an odd-length state
        # does not fit that pairing, so it is rendered whole as "position"
        # with velocity reported as zero.
        pos = state[0::2] if len(state) % 2 == 0 else state
        vel = state[1::2] if len(state) % 2 == 0 else np.zeros(1)
        table.add_row(
            str(t.id),
            getattr(t.status, "value", str(t.status)),
            "(" + ", ".join(f"{p:.1f}" for p in pos) + ")",
            f"{float(np.linalg.norm(vel)):.2f}",
        )
    console.print(table)


def progress_bar(
    iterable: Iterable[Any],
    description: str = "working",
    total: Optional[int] = None,
) -> Iterator[Any]:
    """
    Wrap an iterable in an ASCII progress bar on stderr.

    Yields the items unchanged; the bar completes when iteration ends.

    Notes
    -----
    Uses a pure-text progress display (no Unicode block-bar column):
    rich's default ``BarColumn`` renders with Unicode block characters
    that are not cp1252-encodable, which the console-encoding contract
    forbids.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    has_total = total is not None
    progress = Progress(
        TextColumn("{task.description}"),
        TextColumn(
            "{task.completed}/{task.total}" if has_total else "{task.completed}"
        ),
        TextColumn("{task.percentage:>3.0f}%" if has_total else ""),
        console=Console(stderr=True),
    )
    with progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)
