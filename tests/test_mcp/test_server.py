"""Tests for mnemo.mcp.server startup validation and tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import mnemo.mcp.server as srv
from mnemo.client import MnemoAuthError, MnemoNotFoundError
from mnemo.mcp.server import (
    mnemo_list_shared,
    mnemo_recall,
    mnemo_recall_shared,
    mnemo_remember,
    mnemo_revoke_share,
    mnemo_share,
    mnemo_stats,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

VALID_AGENT_ID = str(uuid4())
OPERATOR_INFO = {"id": str(uuid4()), "name": "test-operator", "agent_count": 1}


def _make_mock_client(
    me_return=None,
    me_side_effect=None,
    get_agent_return=None,
    get_agent_side_effect=None,
) -> AsyncMock:
    """Build an AsyncMock MnemoClient with configurable me/get_agent responses."""
    client = AsyncMock()
    if me_side_effect is not None:
        client.me.side_effect = me_side_effect
    else:
        client.me.return_value = me_return or OPERATOR_INFO

    if get_agent_side_effect is not None:
        client.get_agent.side_effect = get_agent_side_effect
    else:
        client.get_agent.return_value = get_agent_return or {
            "id": VALID_AGENT_ID,
            "name": "test-agent",
        }

    return client


# ── Fixtures (used by later task tests) ──────────────────────────────────────


@pytest.fixture
def mock_client():
    """Create a mock MnemoClient and inject it into the server module."""
    client = AsyncMock()
    original = srv._client
    srv._client = client
    yield client
    srv._client = original


@pytest.fixture
def with_default_agent():
    original = srv.DEFAULT_AGENT_ID
    srv.DEFAULT_AGENT_ID = VALID_AGENT_ID
    yield VALID_AGENT_ID
    srv.DEFAULT_AGENT_ID = original


@pytest.fixture
def without_default_agent():
    original = srv.DEFAULT_AGENT_ID
    srv.DEFAULT_AGENT_ID = None
    yield
    srv.DEFAULT_AGENT_ID = original


# ── Startup validation tests ─────────────────────────────────────────────────


class TestStartupValidation:
    """Tests for _validate_startup()."""

    async def test_startup_no_agent_key(self):
        orig_key, orig_url = srv.AGENT_KEY, srv.BASE_URL
        try:
            srv.AGENT_KEY = ""
            srv.BASE_URL = "http://localhost:8000"
            with pytest.raises(SystemExit):
                await srv._validate_startup()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url

    async def test_startup_no_base_url(self):
        orig_key, orig_url = srv.AGENT_KEY, srv.BASE_URL
        try:
            srv.AGENT_KEY = "mnemo_ag_test"
            srv.BASE_URL = ""
            with pytest.raises(SystemExit):
                await srv._validate_startup()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url

    async def test_startup_invalid_default_agent(self):
        orig_key, orig_url, orig_agent = srv.AGENT_KEY, srv.BASE_URL, srv.DEFAULT_AGENT_ID
        try:
            srv.AGENT_KEY = "mnemo_ag_test"
            srv.BASE_URL = "http://localhost:8000"
            srv.DEFAULT_AGENT_ID = VALID_AGENT_ID

            mock_client = AsyncMock()
            mock_client.stats.side_effect = MnemoNotFoundError("agent not found")

            with patch.object(srv, "MnemoClient", return_value=mock_client):
                with pytest.raises(SystemExit):
                    await srv._validate_startup()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url
            srv.DEFAULT_AGENT_ID = orig_agent

    async def test_startup_invalid_default_agent_auth_error(self):
        orig_key, orig_url, orig_agent = srv.AGENT_KEY, srv.BASE_URL, srv.DEFAULT_AGENT_ID
        try:
            srv.AGENT_KEY = "mnemo_ag_bad"
            srv.BASE_URL = "http://localhost:8000"
            srv.DEFAULT_AGENT_ID = VALID_AGENT_ID

            mock_client = AsyncMock()
            mock_client.stats.side_effect = MnemoAuthError("invalid key")

            with patch.object(srv, "MnemoClient", return_value=mock_client):
                with pytest.raises(SystemExit):
                    await srv._validate_startup()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url
            srv.DEFAULT_AGENT_ID = orig_agent

    async def test_startup_success_with_default_agent(self):
        orig_key, orig_url, orig_agent = srv.AGENT_KEY, srv.BASE_URL, srv.DEFAULT_AGENT_ID
        try:
            srv.AGENT_KEY = "mnemo_ag_test"
            srv.BASE_URL = "http://localhost:8000"
            srv.DEFAULT_AGENT_ID = VALID_AGENT_ID

            mock_client = AsyncMock()
            mock_client.stats.return_value = {"total_atoms": 0, "active_atoms": 0}

            with patch.object(srv, "MnemoClient", return_value=mock_client):
                result = await srv._validate_startup()

            assert result is mock_client
            mock_client.stats.assert_awaited_once()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url
            srv.DEFAULT_AGENT_ID = orig_agent

    async def test_startup_success_no_default(self):
        orig_key, orig_url, orig_agent = srv.AGENT_KEY, srv.BASE_URL, srv.DEFAULT_AGENT_ID
        try:
            srv.AGENT_KEY = "mnemo_ag_test"
            srv.BASE_URL = "http://localhost:8000"
            srv.DEFAULT_AGENT_ID = None

            mock_client = AsyncMock()
            mock_client.health.return_value = {"status": "ok"}

            with patch.object(srv, "MnemoClient", return_value=mock_client):
                result = await srv._validate_startup()

            assert result is mock_client
            mock_client.health.assert_awaited_once()
        finally:
            srv.AGENT_KEY = orig_key
            srv.BASE_URL = orig_url
            srv.DEFAULT_AGENT_ID = orig_agent


# ── mnemo_remember tests ────────────────────────────────────────────────────


class TestRemember:
    """Tests for the mnemo_remember tool."""

    async def test_remember_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_remember(text="test")
        assert "agent_id is required" in result

    async def test_remember_uses_default_agent(
        self, mock_client, with_default_agent
    ):
        mock_client.remember.return_value = {
            "status": "queued",
            "store_id": str(uuid4()),
        }
        result = await mnemo_remember(text="test")
        assert "queued" in result
        call_kwargs = mock_client.remember.call_args.kwargs
        assert str(call_kwargs["agent_id"]) == VALID_AGENT_ID

    async def test_remember_explicit_overrides_default(
        self, mock_client, with_default_agent
    ):
        other_id = str(uuid4())
        mock_client.remember.return_value = {
            "status": "queued",
            "store_id": str(uuid4()),
        }
        result = await mnemo_remember(text="test", agent_id=other_id)
        call_kwargs = mock_client.remember.call_args.kwargs
        assert str(call_kwargs["agent_id"]) == other_id
        assert str(call_kwargs["agent_id"]) != VALID_AGENT_ID

    async def test_remember_invalid_uuid(self, mock_client):
        result = await mnemo_remember(text="test", agent_id="not-a-uuid")
        assert "Invalid agent_id" in result or "valid UUID" in result

    async def test_remember_nonexistent_agent(self, mock_client):
        mock_client.remember.side_effect = MnemoNotFoundError("not found")
        agent_id = str(uuid4())
        result = await mnemo_remember(text="test", agent_id=agent_id)
        assert "not found" in result

    async def test_remember_wrong_operator(self, mock_client):
        mock_client.remember.side_effect = MnemoAuthError("Permission denied")
        agent_id = str(uuid4())
        result = await mnemo_remember(text="test", agent_id=agent_id)
        assert "not owned" in result

    async def test_remember_success(self, mock_client):
        store_id = str(uuid4())
        mock_client.remember.return_value = {
            "status": "queued",
            "store_id": store_id,
        }
        agent_id = str(uuid4())
        result = await mnemo_remember(text="test", agent_id=agent_id)
        assert "queued" in result
        assert store_id in result

    async def test_remember_connection_error(self, mock_client):
        mock_client.remember.side_effect = ConnectionError("refused")
        agent_id = str(uuid4())
        result = await mnemo_remember(text="test", agent_id=agent_id)
        assert "Error" in result

    async def test_remember_generic_exception_does_not_leak(self, mock_client):
        mock_client.remember.side_effect = RuntimeError(
            "SECRET: /internal/path/db.sqlite3"
        )
        agent_id = str(uuid4())
        result = await mnemo_remember(text="test", agent_id=agent_id)
        assert "SECRET" not in result
        assert "db.sqlite3" not in result
        assert "unexpected error" in result


# ── mnemo_recall tests ─────────────────────────────────────────────────────


class TestRecall:
    """Tests for the mnemo_recall tool."""

    async def test_recall_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_recall(query="test")
        assert "agent_id is required" in result

    async def test_recall_uses_default_agent(
        self, mock_client, with_default_agent
    ):
        mock_client.recall.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "Redis is fast",
                    "confidence_effective": 0.8,
                    "relevance_score": 0.92,
                }
            ],
            "expanded_atoms": [],
            "total_retrieved": 1,
        }
        result = await mnemo_recall(query="redis performance")
        assert "Redis" in result
        call_kwargs = mock_client.recall.call_args.kwargs
        assert str(call_kwargs["agent_id"]) == VALID_AGENT_ID

    async def test_recall_empty(self, mock_client, with_default_agent):
        mock_client.recall.return_value = {
            "atoms": [],
            "expanded_atoms": [],
            "total_retrieved": 0,
        }
        result = await mnemo_recall(query="nonexistent topic")
        assert "No relevant memories found" in result

    async def test_recall_safety_frame(self, mock_client, with_default_agent):
        mock_client.recall.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "Some fact",
                    "confidence_effective": 0.8,
                    "relevance_score": 0.9,
                }
            ],
            "expanded_atoms": [],
            "total_retrieved": 1,
        }
        result = await mnemo_recall(query="test")
        assert result.startswith("[Retrieved memories")
        assert result.endswith("[End retrieved memories]")

    async def test_recall_confidence_labels(
        self, mock_client, with_default_agent
    ):
        mock_client.recall.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "High conf fact",
                    "confidence_effective": 0.9,
                    "relevance_score": 0.9,
                },
                {
                    "atom_type": "episodic",
                    "text_content": "Moderate conf fact",
                    "confidence_effective": 0.5,
                    "relevance_score": 0.7,
                },
                {
                    "atom_type": "procedural",
                    "text_content": "Low conf fact",
                    "confidence_effective": 0.2,
                    "relevance_score": 0.5,
                },
            ],
            "expanded_atoms": [],
            "total_retrieved": 3,
        }
        result = await mnemo_recall(query="test")
        assert "high" in result
        assert "moderate" in result
        assert "low" in result

    async def test_recall_with_expanded_atoms(
        self, mock_client, with_default_agent
    ):
        mock_client.recall.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "Primary fact",
                    "confidence_effective": 0.8,
                    "relevance_score": 0.9,
                }
            ],
            "expanded_atoms": [
                {
                    "atom_type": "episodic",
                    "text_content": "Related fact",
                    "relevance_score": 0.6,
                }
            ],
            "total_retrieved": 2,
        }
        result = await mnemo_recall(query="test")
        assert "Primary fact" in result
        assert "Related fact" in result
        assert "Related" in result

    async def test_recall_invalid_uuid(self, mock_client):
        result = await mnemo_recall(query="test", agent_id="bad-uuid")
        assert "Invalid agent_id" in result or "valid UUID" in result

    async def test_recall_clamps_max_results(self, mock_client, with_default_agent):
        mock_client.recall.return_value = {
            "atoms": [], "expanded_atoms": [], "total_retrieved": 0,
        }
        await mnemo_recall(query="test", max_results=999)
        assert mock_client.recall.call_args.kwargs["max_results"] == 50

        await mnemo_recall(query="test", max_results=-5)
        assert mock_client.recall.call_args.kwargs["max_results"] == 1

    async def test_recall_clamps_min_similarity(self, mock_client, with_default_agent):
        mock_client.recall.return_value = {
            "atoms": [], "expanded_atoms": [], "total_retrieved": 0,
        }
        await mnemo_recall(query="test", min_similarity=5.0)
        assert mock_client.recall.call_args.kwargs["min_similarity"] == 1.0

        await mnemo_recall(query="test", min_similarity=-1.0)
        assert mock_client.recall.call_args.kwargs["min_similarity"] == 0.0

    async def test_recall_clamps_similarity_drop_threshold(
        self, mock_client, with_default_agent
    ):
        mock_client.recall.return_value = {
            "atoms": [], "expanded_atoms": [], "total_retrieved": 0,
        }
        await mnemo_recall(query="test", similarity_drop_threshold=2.0)
        assert mock_client.recall.call_args.kwargs["similarity_drop_threshold"] == 1.0

    async def test_recall_clamps_max_total_tokens(
        self, mock_client, with_default_agent
    ):
        mock_client.recall.return_value = {
            "atoms": [], "expanded_atoms": [], "total_retrieved": 0,
        }
        await mnemo_recall(query="test", max_total_tokens=99999)
        assert mock_client.recall.call_args.kwargs["max_total_tokens"] == 10000

    async def test_recall_generic_exception_does_not_leak(self, mock_client):
        mock_client.recall.side_effect = RuntimeError("SENSITIVE_INFO")
        agent_id = str(uuid4())
        result = await mnemo_recall(query="test", agent_id=agent_id)
        assert "SENSITIVE_INFO" not in result
        assert "unexpected error" in result


# ── mnemo_stats tests ──────────────────────────────────────────────────────


class TestStats:
    """Tests for the mnemo_stats tool."""

    async def test_stats_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_stats()
        assert "agent_id is required" in result

    async def test_stats_uses_default_agent(
        self, mock_client, with_default_agent
    ):
        mock_client.stats.return_value = {
            "total_atoms": 10,
            "active_atoms": 8,
            "atoms_by_type": {"semantic": 5, "episodic": 3},
            "arc_atoms": 1,
            "avg_effective_confidence": 0.72,
            "total_edges": 4,
            "active_views": 1,
            "granted_capabilities": 0,
            "received_capabilities": 0,
        }
        result = await mnemo_stats()
        assert "Total memories" in result
        assert "10" in result
        call_kwargs = mock_client.stats.call_args.kwargs
        assert str(call_kwargs["agent_id"]) == VALID_AGENT_ID

    async def test_stats_success(self, mock_client):
        mock_client.stats.return_value = {
            "total_atoms": 5,
            "active_atoms": 5,
            "atoms_by_type": {"semantic": 3, "procedural": 2},
            "avg_effective_confidence": 0.65,
        }
        agent_id = str(uuid4())
        result = await mnemo_stats(agent_id=agent_id)
        assert "Total memories: 5" in result
        assert "active: 5" in result
        assert "Avg confidence" in result
        assert "65%" in result

    async def test_stats_not_found(self, mock_client):
        mock_client.stats.side_effect = MnemoNotFoundError("not found")
        agent_id = str(uuid4())
        result = await mnemo_stats(agent_id=agent_id)
        assert "not found" in result


# ── mnemo_share tests ─────────────────────────────────────────────────────


class TestShare:
    """Tests for the mnemo_share tool."""

    async def test_share_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_share(query="test", share_with="bob:org")
        assert "agent_id is required" in result

    async def test_share_resolve_address_not_found(
        self, mock_client, with_default_agent
    ):
        mock_client.resolve_address.side_effect = MnemoNotFoundError("not found")
        result = await mnemo_share(query="test", share_with="bob:org")
        assert "not found" in result

    async def test_share_success(self, mock_client, with_default_agent):
        grantee_uuid = str(uuid4())
        view_id = str(uuid4())
        cap_id = str(uuid4())

        mock_client.resolve_address.return_value = grantee_uuid
        mock_client.create_view.return_value = {
            "id": view_id,
            "atom_count": 3,
        }
        mock_client.grant.return_value = {"id": cap_id}

        result = await mnemo_share(query="python tips", share_with="bob:org")
        assert "Shared 3 memories" in result
        assert view_id in result
        assert cap_id in result
        assert "bob:org" in result

    async def test_share_with_name(self, mock_client, with_default_agent):
        grantee_uuid = str(uuid4())
        view_id = str(uuid4())

        mock_client.resolve_address.return_value = grantee_uuid
        mock_client.create_view.return_value = {"id": view_id, "atom_count": 1}
        mock_client.grant.return_value = {"id": str(uuid4())}

        result = await mnemo_share(
            query="test", share_with="bob:org", name="my-view"
        )
        assert "my-view" in result

    async def test_share_create_view_auth_error(
        self, mock_client, with_default_agent
    ):
        mock_client.resolve_address.return_value = str(uuid4())
        mock_client.create_view.side_effect = MnemoAuthError("denied")
        result = await mnemo_share(query="test", share_with="bob:org")
        assert "Error" in result

    async def test_share_invalid_uuid(self, mock_client):
        result = await mnemo_share(
            query="test", share_with="bob:org", agent_id="bad"
        )
        assert "Invalid agent_id" in result or "valid UUID" in result


# ── mnemo_list_shared tests ──────────────────────────────────────────────


class TestListShared:
    """Tests for the mnemo_list_shared tool."""

    async def test_list_shared_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_list_shared()
        assert "agent_id is required" in result

    async def test_list_shared_empty(self, mock_client, with_default_agent):
        mock_client.list_shared_views.return_value = []
        result = await mnemo_list_shared()
        assert "No shared views available" in result

    async def test_list_shared_success(self, mock_client, with_default_agent):
        mock_client.list_shared_views.return_value = [
            {
                "name": "python-tips",
                "source_address": "alice:org",
                "description": "Python knowledge",
                "atom_count": 5,
                "granted_at": "2026-03-10T12:00:00Z",
            },
        ]
        result = await mnemo_list_shared()
        assert "python-tips" in result
        assert "alice:org" in result
        assert "Python knowledge" in result

    async def test_list_shared_not_found(self, mock_client):
        mock_client.list_shared_views.side_effect = MnemoNotFoundError("not found")
        agent_id = str(uuid4())
        result = await mnemo_list_shared(agent_id=agent_id)
        assert "not found" in result

    async def test_list_shared_auth_error(self, mock_client):
        mock_client.list_shared_views.side_effect = MnemoAuthError("denied")
        agent_id = str(uuid4())
        result = await mnemo_list_shared(agent_id=agent_id)
        assert "not owned" in result


# ── mnemo_recall_shared tests ────────────────────────────────────────────


class TestRecallShared:
    """Tests for the mnemo_recall_shared tool."""

    async def test_recall_shared_requires_agent_id_when_no_default(
        self, mock_client, without_default_agent
    ):
        result = await mnemo_recall_shared(query="test")
        assert "agent_id is required" in result

    async def test_recall_shared_empty(self, mock_client, with_default_agent):
        mock_client.recall_all_shared.return_value = {"atoms": []}
        result = await mnemo_recall_shared(query="test")
        assert "No relevant shared memories found" in result

    async def test_recall_shared_success(self, mock_client, with_default_agent):
        mock_client.recall_all_shared.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "Redis is an in-memory store",
                    "confidence_effective": 0.85,
                    "relevance_score": 0.91,
                    "source_address": "alice:org",
                },
            ]
        }
        result = await mnemo_recall_shared(query="redis")
        assert "Redis" in result
        assert "alice:org" in result
        assert "high" in result

    async def test_recall_shared_safety_frame(
        self, mock_client, with_default_agent
    ):
        mock_client.recall_all_shared.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "Some fact",
                    "confidence_effective": 0.8,
                    "relevance_score": 0.9,
                    "source_address": "bob:org",
                },
            ]
        }
        result = await mnemo_recall_shared(query="test")
        assert result.startswith("[Shared memories")
        assert result.endswith("[End shared memories]")

    async def test_recall_shared_confidence_labels(
        self, mock_client, with_default_agent
    ):
        mock_client.recall_all_shared.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "High",
                    "confidence_effective": 0.9,
                    "relevance_score": 0.9,
                    "source_address": "a:o",
                },
                {
                    "atom_type": "episodic",
                    "text_content": "Moderate",
                    "confidence_effective": 0.5,
                    "relevance_score": 0.7,
                    "source_address": "b:o",
                },
                {
                    "atom_type": "procedural",
                    "text_content": "Low",
                    "confidence_effective": 0.2,
                    "relevance_score": 0.5,
                    "source_address": "c:o",
                },
            ]
        }
        result = await mnemo_recall_shared(query="test")
        assert "high" in result
        assert "moderate" in result
        assert "low" in result

    async def test_recall_shared_token_budget(
        self, mock_client, with_default_agent
    ):
        # Each atom text ~25 chars => ~6 tokens. Budget of 10 should include
        # at most 2 atoms.
        mock_client.recall_all_shared.return_value = {
            "atoms": [
                {
                    "atom_type": "semantic",
                    "text_content": "A" * 40,
                    "confidence_effective": 0.8,
                    "relevance_score": 0.9,
                    "source_address": "a:o",
                },
                {
                    "atom_type": "semantic",
                    "text_content": "B" * 40,
                    "confidence_effective": 0.8,
                    "relevance_score": 0.8,
                    "source_address": "b:o",
                },
                {
                    "atom_type": "semantic",
                    "text_content": "C" * 40,
                    "confidence_effective": 0.8,
                    "relevance_score": 0.7,
                    "source_address": "c:o",
                },
            ]
        }
        result = await mnemo_recall_shared(query="test", max_total_tokens=10)
        # With budget=10 tokens, 40 chars / 4 = 10 tokens per atom.
        # First atom costs 10, budget goes to 0. Second atom would push below 0
        # but filtered is non-empty so it stops.
        assert "A" * 40 in result
        assert "B" * 40 not in result

    async def test_recall_shared_clamps_max_results(
        self, mock_client, with_default_agent
    ):
        mock_client.recall_all_shared.return_value = {"atoms": []}
        await mnemo_recall_shared(query="test", max_results=999)
        assert mock_client.recall_all_shared.call_args.kwargs["max_results"] == 50

        await mnemo_recall_shared(query="test", max_results=0)
        assert mock_client.recall_all_shared.call_args.kwargs["max_results"] == 1

    async def test_recall_shared_clamps_min_similarity(
        self, mock_client, with_default_agent
    ):
        mock_client.recall_all_shared.return_value = {"atoms": []}
        await mnemo_recall_shared(query="test", min_similarity=2.0)
        assert mock_client.recall_all_shared.call_args.kwargs["min_similarity"] == 1.0

    async def test_recall_shared_not_found(self, mock_client):
        mock_client.recall_all_shared.side_effect = MnemoNotFoundError("not found")
        agent_id = str(uuid4())
        result = await mnemo_recall_shared(query="test", agent_id=agent_id)
        assert "not found" in result

    async def test_recall_shared_connection_error(self, mock_client):
        mock_client.recall_all_shared.side_effect = ConnectionError("refused")
        agent_id = str(uuid4())
        result = await mnemo_recall_shared(query="test", agent_id=agent_id)
        assert "Error" in result


# ── mnemo_revoke_share tests ───────────────────────────────────────────


class TestRevokeShare:
    """Tests for the mnemo_revoke_share tool."""

    async def test_revoke_success(self, mock_client, with_default_agent):
        cap_id = str(uuid4())
        view_id = str(uuid4())
        grantee_id = str(uuid4())
        mock_client.revoke_shared_view.return_value = {
            "capability_id": cap_id,
            "view_id": view_id,
            "grantee_id": grantee_id,
            "revoked": True,
            "revoked_at": "2026-03-14T12:00:00Z",
            "was_already_revoked": False,
        }
        result = await mnemo_revoke_share(capability_id=cap_id)
        assert "Shared view revoked" in result
        assert view_id in result
        assert grantee_id in result
        assert "2026-03-14" in result

    async def test_revoke_already_revoked(self, mock_client, with_default_agent):
        cap_id = str(uuid4())
        mock_client.revoke_shared_view.return_value = {
            "capability_id": cap_id,
            "view_id": str(uuid4()),
            "grantee_id": str(uuid4()),
            "revoked": True,
            "revoked_at": "2026-03-13T10:00:00Z",
            "was_already_revoked": True,
        }
        result = await mnemo_revoke_share(capability_id=cap_id)
        assert "already revoked" in result
        assert "2026-03-13" in result

    async def test_revoke_not_found(self, mock_client, with_default_agent):
        mock_client.revoke_shared_view.side_effect = MnemoNotFoundError("not found")
        cap_id = str(uuid4())
        result = await mnemo_revoke_share(capability_id=cap_id)
        assert "not found" in result or "not the grantor" in result

    async def test_revoke_auth_error(self, mock_client, with_default_agent):
        mock_client.revoke_shared_view.side_effect = MnemoAuthError("denied")
        cap_id = str(uuid4())
        result = await mnemo_revoke_share(capability_id=cap_id)
        assert "not owned" in result

    async def test_revoke_default_agent(self, mock_client, with_default_agent):
        cap_id = str(uuid4())
        mock_client.revoke_shared_view.return_value = {
            "capability_id": cap_id,
            "view_id": str(uuid4()),
            "grantee_id": str(uuid4()),
            "revoked": True,
            "revoked_at": "2026-03-14T12:00:00Z",
            "was_already_revoked": False,
        }
        await mnemo_revoke_share(capability_id=cap_id)
        call_kwargs = mock_client.revoke_shared_view.call_args.kwargs
        assert str(call_kwargs["agent_id"]) == VALID_AGENT_ID

    async def test_revoke_invalid_capability_id(self, mock_client, with_default_agent):
        result = await mnemo_revoke_share(capability_id="not-a-uuid")
        assert "invalid capability_id" in result


# ── mnemo_list_shared outbound tests ───────────────────────────────────


class TestListSharedOutbound:
    """Tests for mnemo_list_shared with direction='outbound'."""

    async def test_outbound_empty(self, mock_client, with_default_agent):
        mock_client.list_outbound_capabilities.return_value = []
        result = await mnemo_list_shared(direction="outbound")
        assert "No outbound shares" in result

    async def test_outbound_success(self, mock_client, with_default_agent):
        cap_id = str(uuid4())
        mock_client.list_outbound_capabilities.return_value = [
            {
                "capability_id": cap_id,
                "view_id": str(uuid4()),
                "view_name": "python-tips",
                "grantee_id": str(uuid4()),
                "grantee_address": "bob:org",
                "permissions": ["read"],
                "revoked": False,
                "revoked_at": None,
                "granted_at": "2026-03-12T10:00:00Z",
            },
        ]
        result = await mnemo_list_shared(direction="outbound")
        assert "python-tips" in result
        assert "bob:org" in result
        assert cap_id in result
        assert "active" in result

    async def test_outbound_revoked_status(self, mock_client, with_default_agent):
        mock_client.list_outbound_capabilities.return_value = [
            {
                "capability_id": str(uuid4()),
                "view_id": str(uuid4()),
                "view_name": "old-share",
                "grantee_id": str(uuid4()),
                "grantee_address": "alice:org",
                "permissions": ["read"],
                "revoked": True,
                "revoked_at": "2026-03-13T10:00:00Z",
                "granted_at": "2026-03-10T10:00:00Z",
            },
        ]
        result = await mnemo_list_shared(direction="outbound")
        assert "revoked" in result
        assert "old-share" in result
