"""mnemo v2 Phase 1: MCP tool behaviour (mocked client)."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

import mnemo.mcp.server as srv
from mnemo.client import MnemoClient, MnemoStoreFailedError
from mnemo.mcp.server import (
    REMEMBER_DESCRIPTION, mnemo_forget, mnemo_recall, mnemo_recent, mnemo_remember, mnemo_store_status,
)

SELF = str(uuid4())
A1, A2, STORE = str(uuid4()), str(uuid4()), str(uuid4())


@pytest.fixture
def client():
    c = AsyncMock()
    c.whoami.return_value = {"id": SELF, "name": "clio", "status": "active"}
    orig_client, orig_default = srv._client, srv.DEFAULT_AGENT_ID
    srv._client, srv.DEFAULT_AGENT_ID, srv._self_agent_id = c, None, None
    yield c
    srv._client, srv.DEFAULT_AGENT_ID, srv._self_agent_id = orig_client, orig_default, None


STORED = {
    "status": "complete", "store_id": STORE, "source_id": str(uuid4()),
    "atoms_created": 1, "duplicates_merged": 1,
    "atoms": [
        {"id": A1, "atom_type": "episodic", "text_content": "Tom moved the release to 2026-12-03.",
         "remembered_on": "2026-11-20T00:00:00Z", "merged_into_existing": False, "flags": []},
        {"id": A2, "atom_type": "semantic", "text_content": "The VXUS sleeve is rebalanced monthly.",
         "remembered_on": None, "merged_into_existing": True, "flags": ["abbreviation_altered"]},
    ],
    "warnings": ["input split into 2 parts"],
}


def test_remember_description_text():
    assert "Abbreviations and codes are stored exactly as written. Spell them out in the text if the meaning matters." in REMEMBER_DESCRIPTION


async def test_remember_lists_atoms_with_ids_and_flags(client):
    client.remember.return_value = STORED
    out = await mnemo_remember(text="x", speaker="Tom", addressee="Clio", channel="zulip",
                               origin_class="user_stated", remembered_on="2026-11-20")
    lines = out.splitlines()
    assert lines[0] == f"Stored 1 new, 1 merged (store_id {STORE})."
    assert lines[1] == f"- {A1} [episodic 2026-11-20] Tom moved the release to 2026-12-03."
    assert lines[2] == f"- {A2} [semantic, merged] The VXUS sleeve is rebalanced monthly.  ⚑ abbreviation_altered"
    assert "(note: input split into 2 parts)" in out
    kw = client.remember.call_args.kwargs
    assert kw["agent_id"] == SELF  # defaulted from /agents/me
    assert (kw["speaker"], kw["addressee"], kw["channel"], kw["origin_class"]) == ("Tom", "Clio", "zulip", "user_stated")


async def test_whoami_cached(client):
    client.remember.return_value = STORED
    await mnemo_remember(text="x")
    await mnemo_remember(text="y")
    client.whoami.assert_awaited_once()


@pytest.mark.parametrize("ref", ["clio", "clio:tom.inforge", str(uuid4())])
async def test_agent_name_or_address_passed_through(client, ref):
    client.remember.return_value = STORED
    await mnemo_remember(text="x", agent_id=ref)
    assert client.remember.call_args.kwargs["agent_id"] == ref
    client.whoami.assert_not_awaited()


async def test_remember_failure_keeps_text_message(client):
    client.remember.side_effect = MnemoStoreFailedError(STORE, "src", "provider timeout", 502)
    out = await mnemo_remember(text="x")
    assert out.startswith("Not stored: provider timeout.")
    assert STORE in out and "retry=True" in out


async def test_remember_bad_origin_class(client):
    assert "origin_class must be one of" in await mnemo_remember(text="x", origin_class="rumour")


async def test_recall_full_maps_to_truncated_with_cap_and_shows_ids(client):
    client.recall.return_value = {"atoms": [
        {"id": A1, "atom_type": "episodic", "text_content": "Full text.", "remembered_on": "2026-11-20",
         "confidence_effective": 0.9, "relevance_score": 0.8}], "expanded_atoms": []}
    out = await mnemo_recall(query="q", since="2026-11-01", until="2026-12-01", atom_types=["episodic"])
    kw = client.recall.call_args.kwargs
    assert kw["verbosity"] == "truncated" and kw["max_content_chars"] == 600
    assert (kw["since"], kw["until"], kw["atom_types"]) == ("2026-11-01", "2026-12-01", ["episodic"])
    assert f"- {A1} [episodic 2026-11-20, high conf, 0.80] Full text." in out


async def test_recall_summary_and_validation(client):
    client.recall.return_value = {"atoms": [], "expanded_atoms": []}
    await mnemo_recall(query="q", verbosity="summary")
    kw = client.recall.call_args.kwargs
    assert kw["verbosity"] == "summary" and kw["max_content_chars"] is None
    assert "verbosity must be" in await mnemo_recall(query="q", verbosity="loud")
    assert "atom_types must be" in await mnemo_recall(query="q", atom_types=["relational"])


async def test_recent(client):
    client.recent.return_value = {"atoms": [
        {"id": A1, "atom_type": "episodic", "text_content": "Newest.", "remembered_on": "2026-11-21"}],
        "total_returned": 1}
    out = await mnemo_recent(limit=500, order_by="event", atom_types=["episodic"])
    kw = client.recent.call_args.kwargs
    assert kw["limit"] == 100 and kw["order_by"] == "event" and kw["agent_id"] == SELF
    assert f"- {A1} [episodic 2026-11-21] Newest." in out
    assert "order_by must be" in await mnemo_recent(order_by="random")


async def test_forget(client):
    out = await mnemo_forget(atom_id=A1)
    assert out == f"Forgot memory {A1}."
    assert str(client.delete_atom.call_args.kwargs["atom_id"]) == A1
    assert "invalid atom_id" in await mnemo_forget(atom_id="nope")


async def test_store_status_and_retry(client):
    client.store_status.return_value = {"status": "failed", "error": "provider timeout", "atoms": []}
    out = await mnemo_store_status(store_id=STORE)
    assert out.startswith(f"Store {STORE}: failed — provider timeout") and "retry=True" in out
    client.retry_store.return_value = STORED
    out = await mnemo_store_status(store_id=STORE, retry=True)
    assert out.startswith("Stored 1 new, 1 merged")


# ── MnemoClient against a mocked transport ─────────────────────────────────

async def test_client_remember_body_timeout_and_failure():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(502, json={"status": "failed", "store_id": STORE, "source_id": "s",
                                         "error": "provider timeout"})

    c = MnemoClient("http://mnemo.test", agent_key="k")
    c._agent_http = httpx.AsyncClient(base_url="http://mnemo.test", transport=httpx.MockTransport(handler))
    with pytest.raises(MnemoStoreFailedError) as exc:
        await c.remember("clio", "text", speaker="Tom", origin_class="user_stated")
    await c.close()
    assert exc.value.store_id == STORE and exc.value.error == "provider timeout"
    assert seen["path"] == "/v1/agents/clio/remember"
    assert b'"speaker":"Tom"' in seen["body"].replace(b" ", b"") and b'"wait":true' in seen["body"].replace(b" ", b"")
    assert seen["timeout"]["read"] == 120.0
