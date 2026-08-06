from __future__ import annotations

from mirage.schemas import VariableRole


_CONTEXT = {
    "LDCOUT",
    "TOTAIRFL",
    "TOTFUELFL",
    "TOTFWFL",
    "SPROUTT",
    "MSTEAMFL",
    "MSTMPRESS",
    "FWFUELDIV",
}
_OUTPUT = {"LBA30CT", "LBB30CT", "LBB30CP"}
_PROCESS_PREFIXES = ("HAH79", "HAH82", "HAH84", "HAD83", "HAC71")
_ACTUATOR_PREFIXES = ("LAB72CF0A", "HHA01AA19X", "HHA02AA19X", "HHA03AA19X", "HHA04AA19X", "HBB12AA1")


def infer_boiler_role(name: str) -> VariableRole:
    """Initial expert-review mapping from the implementation plan; UNKNOWN is intentional."""
    normalized = name.strip().upper()
    if normalized in _CONTEXT:
        return VariableRole.CONTEXT
    if normalized in _OUTPUT:
        return VariableRole.OUTPUT
    if normalized == "LBB30CT306" or normalized.startswith(_PROCESS_PREFIXES):
        return VariableRole.PROCESS
    if normalized.startswith(_ACTUATOR_PREFIXES):
        return VariableRole.ACTUATOR_COMMAND
    return VariableRole.UNKNOWN

