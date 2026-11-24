"""mnemo v2 Phase 2: supersede, undo, conflicts, resolves, relation output (mocked client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest

import mnemo.mcp.server as srv
from mnemo.client import MnemoClient
from mnemo.mcp.server import mnemo_conflicts, mnemo_recall, mnemo_remember, mnemo_supersede, mnemo_undo_relation

SELF, OLD, NEW, EDGE, STORE = (str(uuid4()) for _ in range(5))


@pytest.fixture
def client():
    c = AsyncMock()
    c.whoami.return_value = {"id": SELF, "name": "clio", "status": "active"}
    orig_client, orig_default = srv._client, srv.DEFAULT_AGENT_ID
    srv._client, srv.DEFAULT_AGENT_ID, srv._self_agent_id = c, None, None
    yield c
    srv._client, srv.DEFAULT_AGENT_ID, srv._self_agent_id = orig_client, orig_default, None


def _status_error(code, detail):
    resp = MagicMock(status_code=code)
    resp.json.return_value = {"detail": detail}
    return httpx.HTTPStatusError("x", request=MagicMock(), response=resp)


async def test_supersede_by_atom(client):
    client.supersede.return_value = {"edge_id": EDGE, "old_atom_id": OLD, "by_atom_id": NEW, "status": "superseded"}
    out = await mnemo_supersede(atom_id=OLD, reason="moved", by_atom_id=NEW)
    assert out == f'Memory {OLD} is now superseded by {NEW}. Undo with mnemo_undo_relation(edge_id="{EDGE}").'
    kw = client.supersede.call_args.kwargs
    assert kw["agent_id"] == SELF and kw["by_atom_id"] == NEW and kw["text"] is None and kw["reason"] == "moved"


@pytest.mark.parametrize("kw", [{}, {"by_atom_id": NEW, "text": "x"}])
async def test_supersede_needs_exactly_one(client, kw):
    assert await mnemo_supersede(atom_id=OLD, reason="r", **kw) == "Error: give exactly one of by_atom_id or text"
    client.supersede.assert_not_called()


async def test_supersede_409_detail(client):
    client.supersede.side_effect = _status_error(409, "atom is superseded; only an active atom can be superseded")
    assert await mnemo_supersede(atom_id=OLD, reason="r", text="new memory") == \
        "Error: atom is superseded; only an active atom can be superseded"


async def test_undo(client):
    client.undo_relation.return_value = {"edge_id": EDGE, "undone": "replaces", "restored_atom_id": OLD}
    assert await mnemo_undo_relation(edge_id=EDGE) == f"Undid replaces: memory {OLD} is active again."
    assert await mnemo_undo_relation(edge_id="nope") == "Error: invalid edge_id 'nope' (expected a UUID)"


async def test_conflicts_lists_and_tells_how_to_settle(client):
    a, b, cid = str(uuid4()), str(uuid4()), str(uuid4())
    client.conflicts.return_value = [{
        "id": cid, "opened_at": "2026-11-24T00:00:00Z", "resolved_at": None, "resolved_by_atom_id": None,
        "origins_differ": True,
        "atoms": [{"id": a, "text_content": "The pump draws 40 W.", "kind": "state", "confidence_expected": 0.89,
                   "origin_class": "measurement"},
                  {"id": b, "text_content": "The pump draws 55 W.", "kind": None, "confidence_expected": 0.6,
                   "origin_class": "agent_inference"}]}]
    out = (await mnemo_conflicts()).splitlines()
    assert out[0] == f"Conflict {cid} (open, different origins):"
    assert out[1] == f"  - {a} [state, measurement, confidence 0.89] The pump draws 40 W."
    assert out[2] == f"  - {b} [unknown, agent_inference, confidence 0.60] The pump draws 55 W."
    assert out[3] == f'  settle with mnemo_remember(text=..., resolves=["{a}", "{b}"])'
    client.conflicts.return_value = []
    assert await mnemo_conflicts() == "No open conflicts."


async def test_remember_resolves_and_relation_lines(client):
    client.remember.return_value = {
        "status": "complete", "store_id": STORE, "atoms_created": 1, "duplicates_merged": 0,
        "atoms": [{"id": NEW, "atom_type": "episodic", "text_content": "Baseline is 54.", "remembered_on": None,
                   "merged_into_existing": False, "flags": [], "kind": "state",
                   "relations": [{"relation": "replaces", "target_atom_id": OLD, "edge_id": EDGE,
                                  "decision_id": str(uuid4()), "mode": "enforce"},
                                 {"relation": "refines", "target_atom_id": OLD, "edge_id": None,
                                  "decision_id": str(uuid4()), "mode": "shadow"}]}],
        "relations_pending": [NEW], "warnings": [],
    }
    out = (await mnemo_remember(text="x", resolves=[OLD, NEW])).splitlines()
    assert out[1] == f"- {NEW} [episodic, state] Baseline is 54."
    assert out[2] == f'    replaces {OLD} (undo: mnemo_undo_relation(edge_id="{EDGE}"))'
    assert "shadow" not in "\n".join(out) and len(out) == 4
    assert out[3] == "(note: relations still being decided for 1 atom(s))"
    assert client.remember.call_args.kwargs["resolves"] == [OLD, NEW]
    assert (await mnemo_remember(text="x", resolves=[OLD])).startswith("Error: resolves takes exactly two")


async def test_recall_marks_hidden_and_conflict_partner(client):
    client.recall.return_value = {"atoms": [
        {"id": OLD, "atom_type": "episodic", "text_content": "Old.", "remembered_on": None, "status": "superseded",
         "hidden_by": {"relation": "replaces", "by_atom_id": NEW, "edge_id": EDGE}},
        {"id": NEW, "atom_type": "episodic", "text_content": "Other side.", "remembered_on": None,
         "match_type": "conflict_partner", "via": OLD, "conflict_id": str(uuid4())},
    ], "total_retrieved": 2}
    out = await mnemo_recall(query="q")
    assert f"[episodic, superseded by {NEW}, undo edge {EDGE}," in out
    assert f"[episodic, conflicts with {OLD}," in out


async def test_client_methods_hit_the_right_paths():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.url.query.decode(), request.content))
        if request.url.path.endswith("/conflicts"):
            return httpx.Response(200, json={"conflicts": []})
        return httpx.Response(201 if "supersede" in request.url.path else 200, json={"ok": True})
    c = MnemoClient(base_url="http://mnemo.test", agent_key="k")
    c._agent_http = httpx.AsyncClient(base_url="http://mnemo.test", transport=httpx.MockTransport(handler))
    await c.supersede(SELF, OLD, reason="r", by_atom_id=NEW)
    await c.undo_relation(EDGE)
    await c.conflicts(SELF, open=False, limit=5)
    await c.close()
    assert [(m, p) for m, p, _, _ in seen] == [
        ("POST", f"/v1/agents/{SELF}/atoms/{OLD}/supersede"), ("POST", f"/v1/edges/{EDGE}/undo"),
        ("GET", f"/v1/agents/{SELF}/conflicts")]
    assert seen[2][2] == "open=false&limit=5"
