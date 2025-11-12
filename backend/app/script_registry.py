from __future__ import annotations

from typing import Dict, List

from .models import ScriptMetadata
from .scripts.base import ScriptDefinition
from .scripts.extract_expected_utterance_parts import (
    SCRIPT_DEFINITION as EXTRACT_EXPECTED_PARTS,
)
from .scripts.get_sop_pipeline import SCRIPT_DEFINITION as SOP_PIPELINE
from .scripts.process_client_cases import SCRIPT_DEFINITION as CLIENT_CASE_PREPROCESS
from .scripts.detect_force_compliance import SCRIPT_DEFINITION as FORCE_COMPLIANCE
from .scripts.select_test_cases import SCRIPT_DEFINITION as CLIENT_CASE_SELECTION
from .scripts.process_waxu_badcase import SCRIPT_DEFINITION as WAXU_BADCASE
from .scripts.update_latest_customer_message import SCRIPT_DEFINITION as UPDATE_LATEST_CUSTOMER
from .scripts.map_sop_ids import SCRIPT_DEFINITION as MAP_SOP_IDS
from .scripts.merge_excel_files import SCRIPT_DEFINITION as MERGE_EXCELS
from .scripts.process_zlkt_reflow import SCRIPT_DEFINITION as ZLKT_REFLOW
from .scripts.zlkt_sop_reference_matcher import SCRIPT_DEFINITION as ZLKT_SOP_MATCHER

_SCRIPT_DEFINITIONS: Dict[str, ScriptDefinition] = {
    definition.metadata.id: definition
    for definition in [
        UPDATE_LATEST_CUSTOMER,
        SOP_PIPELINE,
        EXTRACT_EXPECTED_PARTS,
        CLIENT_CASE_PREPROCESS,
        CLIENT_CASE_SELECTION,
        FORCE_COMPLIANCE,
        MAP_SOP_IDS,
        MERGE_EXCELS,
        ZLKT_REFLOW,
        ZLKT_SOP_MATCHER,
        WAXU_BADCASE,
    ]
}


def list_scripts() -> List[ScriptMetadata]:
    return [definition.metadata for definition in _SCRIPT_DEFINITIONS.values()]


def get_script(script_id: str) -> ScriptDefinition | None:
    return _SCRIPT_DEFINITIONS.get(script_id)
