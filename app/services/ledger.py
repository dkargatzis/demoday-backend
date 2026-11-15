"""Tool ledger: every Gemini and fal.ai call a job makes, with latency and outcome.

The orchestrator binds the running job to the task context; any code on that task (or tasks created from it) that
wraps a call in `ledger.call(...)` adds a row to `job.tool_calls`. Rows are never written once the job is terminal,
so the `done` event, the last status snapshot and `GET /api/jobs/{id}` stay identical. Without `bind` it is a no-op.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Dict, Optional

from app.models.schemas import JobStatus, ToolCall

_JOB: ContextVar[Optional[JobStatus]] = ContextVar("demoday_job", default=None)
TOOL_NAMES = {"gemini": "Gemini", "fal": "fal.ai"}
TERMINAL = ("completed", "failed")


def bind(job: Optional[JobStatus]) -> None:
    """Makes `job` the ledger target for the current task and the tasks it creates from now on."""
    _JOB.set(job)


def current() -> Optional[JobStatus]:
    return _JOB.get()


@asynccontextmanager
async def call(tool: str, purpose: str, model: str) -> AsyncIterator[Optional[ToolCall]]:
    """Records one tool call around the wrapped block; exceptions (including cancellation) are re-raised untouched."""
    job = _JOB.get()
    row: Optional[ToolCall] = None
    if job is not None and job.status not in TERMINAL:
        try:
            row = ToolCall(tool=tool, purpose=purpose, model=model, started_at=time.time())
            job.tool_calls.append(row)
        except Exception:  # the ledger never breaks a call
            row = None
    try:
        yield row
    except asyncio.CancelledError:
        _finish(job, row, ok=False, note="cancelled")
        raise
    except BaseException as exc:
        _finish(job, row, ok=False, note=_short(exc))
        raise
    else:
        _finish(job, row, ok=True)


def _finish(job: Optional[JobStatus], row: Optional[ToolCall], ok: bool, note: str = "") -> None:
    if row is None or job is None or job.status in TERMINAL:
        return
    row.ended_at = time.time()
    row.ms = int((row.ended_at - row.started_at) * 1000)
    row.ok = ok
    row.note = note


def seal(job: JobStatus) -> None:
    """Closes rows still open when the job finishes (call right before the status turns terminal)."""
    now = time.time()
    for row in job.tool_calls:
        if row.ended_at is None:
            row.ended_at = now
            row.ms = int((now - row.started_at) * 1000)
            row.note = row.note or "still running when the job finished"


def summary(job: JobStatus) -> str:
    """'Gemini 5 calls (31.2 s) · fal.ai 9 calls (58.7 s) · 1 fallback' for the job log and the dashboard."""
    totals: Dict[str, Dict[str, float]] = {}
    fallbacks = 0
    for row in job.tool_calls:
        bucket = totals.setdefault(row.tool, {"calls": 0, "ms": 0})
        bucket["calls"] += 1
        bucket["ms"] += row.ms or 0
        if row.ok is False and row.purpose != "warm-up":
            fallbacks += 1
    parts = [f"{TOOL_NAMES.get(tool, tool)} {int(b['calls'])} call{'' if b['calls'] == 1 else 's'} ({b['ms'] / 1000:.1f} s)"
             for tool, b in totals.items()]
    if fallbacks:
        parts.append(f"{fallbacks} fallback{'' if fallbacks == 1 else 's'}")
    return " · ".join(parts) or "no Gemini or fal.ai calls"


def _short(exc: BaseException) -> str:
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:160]
