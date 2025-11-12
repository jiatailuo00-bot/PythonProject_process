from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from starlette.concurrency import run_in_threadpool

from ..models import ScriptMetadata, ScriptRunResponse


RunnerCallable = Callable[[Dict[str, Any]], ScriptRunResponse]


@dataclass(slots=True)
class ScriptDefinition:
    """Connects script metadata with an executable runner."""

    metadata: ScriptMetadata
    runner: RunnerCallable

    async def run(self, params: Dict[str, Any]) -> ScriptRunResponse:
        """Execute the runner inside a thread pool to keep the event loop free."""
        return await run_in_threadpool(self.runner, params)
