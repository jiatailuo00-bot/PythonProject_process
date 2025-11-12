from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ScriptParameter(BaseModel):
    """Describes a single input parameter that a script requires."""

    name: str
    label: str
    type: Literal["string", "number", "boolean", "path", "select"] = "string"
    required: bool = True
    description: Optional[str] = None
    placeholder: Optional[str] = None
    options: Optional[List[str]] = None
    example: Optional[Any] = None


class ScriptMetadata(BaseModel):
    """Human-friendly meta information presented on the UI."""

    id: str
    name: str
    description: str
    category: str = "通用"
    parameters: List[ScriptParameter] = Field(default_factory=list)
    doc_url: Optional[str] = None
    output_description: Optional[str] = None


class ScriptRunRequest(BaseModel):
    """Payload sent from the UI when a script should run."""

    params: Dict[str, Any] = Field(default_factory=dict)


class ScriptRunResponse(BaseModel):
    """Normalized response returned for any script execution."""

    success: bool
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    logs: Optional[str] = None
