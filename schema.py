"""Schema versioning for the structural state files (Wave 1 item 1C).

Problem
-------
The two dict-shaped STRUCTURAL state files — structure_state.json
(dealing_range) and active_obs.json (the slate) — carried no version. Readers
did `json.load` + `isinstance(dict)` only. If a file's SHAPE ever changes (new
code reading an old file, a hand-edited file, a corrupted write), the old reader
silently MIS-PARSES it instead of failing — wrong trend / wrong OBs / bad
alerts. On a money system a silent wrong answer is the worst failure mode.

Fix (one package, signed off by the owner)
------------------------------------------
- stamp(): writers add "schema_version" to the dict before save.
- check(): readers validate it. **MISSING version is treated as the expected
  version** — so every state file written before this change keeps working on
  the first run after deploy (no flag day). Only a MISMATCH (present but != the
  expected version) fails LOUD: it raises SchemaVersionError, which stops the
  scan red in the Actions tab rather than letting it run on a misread file.

Scope: ONLY the two dict-shaped structural state files. The generic JSON
log/gate writers are deliberately untouched — many of them store LISTS (you
can't stamp a version on a JSON array) and their shape is not load-bearing.

Bump CURRENT_VERSION only on a REAL structural schema change, and add a
migration when you do.
"""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_KEY = "schema_version"
CURRENT_VERSION = 1


class SchemaVersionError(Exception):
    """Raised when a state file carries a schema_version != the expected one.

    Fail-loud (owner-approved): a mismatched, present version means the file
    shape may not match this code. Stop, don't silently mis-parse.
    """


def stamp(data: Dict[str, Any], version: int = CURRENT_VERSION) -> Dict[str, Any]:
    """Return `data` with the schema version set (mutates and returns it).

    Pure/safe: only adds one key. Writers call this immediately before dumping.
    """
    if isinstance(data, dict):
        data[SCHEMA_KEY] = version
    return data


def check(data: Any, expected: int = CURRENT_VERSION, name: str = "state") -> Any:
    """Validate the schema version on a loaded dict and return it unchanged.

    Rules:
      - Not a dict  -> return as-is (the caller's own isinstance guard handles
        empties/malformed; versioning only applies to dict state files).
      - MISSING key -> treated as `expected` (v1). No failure: pre-existing
        files have no stamp and must keep working on first run after deploy.
      - PRESENT and == expected -> OK.
      - PRESENT and != expected -> raise SchemaVersionError (FAIL LOUD).
    """
    if not isinstance(data, dict):
        return data
    found = data.get(SCHEMA_KEY)
    if found is None:
        return data  # missing == v1 (deploy-safe)
    if found != expected:
        raise SchemaVersionError(
            f"{name}: schema_version {found!r} != expected {expected!r}. "
            f"This file was written by a different schema; refusing to parse it "
            f"on potentially-mismatched code. Migrate the file or fix the code."
        )
    return data
