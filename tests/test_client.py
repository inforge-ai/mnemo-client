"""Tests for mnemo.client — MnemoClient and MnemoClientSync."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from mnemo.client import (
    MnemoAuthError,
    MnemoClient,
    MnemoClientSync,
    MnemoError,
    MnemoNotFoundError,
    MnemoServerError,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

AGENT_KEY = "mnemo_ag_test123"
OPERATOR_KEY = "mnemo_op_test456"
BASE_URL = "https://api.test.mnemo-ai.com"
AGENT_ID = uuid4()
ATOM_ID = uuid4()
VIEW_ID = uuid4()
GRANTEE_ID = uuid4()
CAP_ID = uuid4()


def _mock_response(status_code: int = 200, json_data=None, text: str = "") -> AsyncMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ── Constructor tests ────────────────────────────────────────────────────────


class TestConstructor:
    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError, match="At least one"):
            MnemoClient(BASE_URL)

    def test_agent_key_only(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY)
        assert c.agent_key == AGENT_KEY
        assert c.operator_key is None

    def test_operator_key_only(self):
        c = MnemoClient(BASE_URL, operator_key=OPERATOR_KEY)
        assert c.operator_key == OPERATOR_KEY
        assert c.agent_key is None

    def test_both_keys(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY, operator_key=OPERATOR_KEY)
        assert c.agent_key == AGENT_KEY
        assert c.operator_key == OPERATOR_KEY

    def test_legacy_api_key_maps_to_operator(self):
        c = MnemoClient(BASE_URL, api_key="legacy_key")
        assert c.operator_key == "legacy_key"

    def test_base_url_trailing_slash_stripped(self):
        c = MnemoClient(BASE_URL + "/", agent_key=AGENT_KEY)
        assert not c.base_url.endswith("/")


# ── _raise_for_status tests ─────────────────────────────────────────────────


class TestRaiseForStatus:
    def setup_method(self):
        self.client = MnemoClient(BASE_URL, agent_key=AGENT_KEY)

    def test_401_raises_auth_error(self):
        resp = _mock_response(401)
        with pytest.raises(MnemoAuthError, match="Invalid or missing"):
            self.client._raise_for_status(resp)

    def test_403_raises_auth_error(self):
        resp = _mock_response(403, text="forbidden")
        with pytest.raises(MnemoAuthError, match="Permission denied"):
            self.client._raise_for_status(resp)

    def test_404_raises_not_found(self):
        resp = _mock_response(404, text="not found")
        with pytest.raises(MnemoNotFoundError):
            self.client._raise_for_status(resp)

    def test_500_raises_server_error(self):
        resp = _mock_response(500, text="internal error")
        with pytest.raises(MnemoServerError, match="500"):
            self.client._raise_for_status(resp)

    def test_502_raises_server_error(self):
        resp = _mock_response(502, text="bad gateway")
        with pytest.raises(MnemoServerError, match="502"):
            self.client._raise_for_status(resp)

    def test_200_passes(self):
        resp = _mock_response(200)
        self.client._raise_for_status(resp)  # should not raise

    def test_exception_hierarchy(self):
        assert issubclass(MnemoAuthError, MnemoError)
        assert issubclass(MnemoNotFoundError, MnemoError)
        assert issubclass(MnemoServerError, MnemoError)


# ── Client property tests ───────────────────────────────────────────────────


class TestClientProperties:
    def test_agent_client_requires_agent_key(self):
        c = MnemoClient(BASE_URL, operator_key=OPERATOR_KEY)
        with pytest.raises(MnemoAuthError, match="agent key"):
            _ = c._agent_client

    def test_operator_client_requires_operator_key(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY)
        with pytest.raises(MnemoAuthError, match="operator key"):
            _ = c._operator_client

    def test_http_property_prefers_agent(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY, operator_key=OPERATOR_KEY)
        assert c.http is c._agent_client

    def test_http_property_falls_back_to_operator(self):
        c = MnemoClient(BASE_URL, operator_key=OPERATOR_KEY)
        assert c.http is c._operator_client


# ── Context manager / close tests ────────────────────────────────────────────


class TestLifecycle:
    async def test_close_closes_both_clients(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY, operator_key=OPERATOR_KEY)
        # Access properties to create underlying httpx clients
        _ = c._agent_client
        _ = c._operator_client
        assert c._agent_http is not None
        assert c._operator_http is not None
        await c.close()
        assert c._agent_http.is_closed
        assert c._operator_http.is_closed

    async def test_context_manager(self):
        async with MnemoClient(BASE_URL, agent_key=AGENT_KEY) as c:
            _ = c._agent_client
        assert c._agent_http.is_closed

    async def test_close_when_no_clients_created(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY)
        await c.close()  # should not raise


# ── Data-plane operation tests ───────────────────────────────────────────────


class TestDataPlaneOps:
    """Test agent-key operations by mocking the httpx client."""

    def setup_method(self):
        self.client = MnemoClient(BASE_URL, agent_key=AGENT_KEY)
        self.mock_http = AsyncMock(spec=httpx.AsyncClient)
        self.client._agent_http = self.mock_http
        self.mock_http.is_closed = False

    async def test_remember(self):
        expected = {"status": "queued", "store_id": "abc123"}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.remember(AGENT_ID, "test memory", ["python"])
        assert result == expected

        call_args = self.mock_http.post.call_args
        assert f"/v1/agents/{AGENT_ID}/remember" in call_args.args[0]
        body = call_args.kwargs["json"]
        assert body["text"] == "test memory"
        assert body["domain_tags"] == ["python"]

    async def test_remember_with_remembered_on(self):
        self.mock_http.post.return_value = _mock_response(
            200, {"status": "queued", "store_id": "abc"}
        )
        await self.client.remember(
            AGENT_ID, "test", remembered_on="2026-03-01T00:00:00Z"
        )
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["remembered_on"] == "2026-03-01T00:00:00Z"

    async def test_recall(self):
        expected = {"atoms": [], "expanded_atoms": [], "total_retrieved": 0}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.recall(AGENT_ID, "search query")
        assert result == expected

        call_args = self.mock_http.post.call_args
        assert f"/v1/agents/{AGENT_ID}/recall" in call_args.args[0]
        body = call_args.kwargs["json"]
        assert body["query"] == "search query"

    async def test_recall_optional_params(self):
        self.mock_http.post.return_value = _mock_response(
            200, {"atoms": [], "expanded_atoms": [], "total_retrieved": 0}
        )
        await self.client.recall(
            AGENT_ID,
            "q",
            similarity_drop_threshold=0.5,
            verbosity="full",
            max_total_tokens=1000,
        )
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["similarity_drop_threshold"] == 0.5
        assert body["verbosity"] == "full"
        assert body["max_total_tokens"] == 1000

    async def test_stats(self):
        expected = {"total_atoms": 10, "active_atoms": 8}
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.stats(AGENT_ID)
        assert result == expected
        assert f"/v1/agents/{AGENT_ID}/stats" in self.mock_http.get.call_args.args[0]

    async def test_store_atom(self):
        expected = {"id": str(uuid4())}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.store_atom(
            AGENT_ID, "semantic", "test content", domain_tags=["test"]
        )
        assert result == expected
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["atom_type"] == "semantic"
        assert body["text_content"] == "test content"

    async def test_get_atom(self):
        expected = {"id": str(ATOM_ID), "text_content": "hello"}
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.get_atom(AGENT_ID, ATOM_ID)
        assert result == expected
        path = self.mock_http.get.call_args.args[0]
        assert str(AGENT_ID) in path
        assert str(ATOM_ID) in path

    async def test_delete_atom(self):
        self.mock_http.delete.return_value = _mock_response(200)
        await self.client.delete_atom(AGENT_ID, ATOM_ID)
        path = self.mock_http.delete.call_args.args[0]
        assert str(ATOM_ID) in path

    async def test_link(self):
        source, target = uuid4(), uuid4()
        expected = {"id": str(uuid4())}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.link(AGENT_ID, source, target, "relates_to")
        assert result == expected
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["source_id"] == str(source)
        assert body["target_id"] == str(target)
        assert body["edge_type"] == "relates_to"
        assert body["weight"] == 1.0

    async def test_create_view(self):
        expected = {"id": str(uuid4()), "atom_count": 3}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.create_view(
            AGENT_ID, "my-view", {"query": "python"}, description="Python tips"
        )
        assert result == expected

    async def test_list_views(self):
        expected = [{"id": "v1", "name": "view1"}]
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.list_views(AGENT_ID)
        assert result == expected

    async def test_export_skill(self):
        expected = {"skill": "data"}
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.export_skill(AGENT_ID, VIEW_ID)
        assert result == expected
        path = self.mock_http.get.call_args.args[0]
        assert "export_skill" in path

    async def test_grant(self):
        expected = {"id": str(uuid4())}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.grant(AGENT_ID, VIEW_ID, GRANTEE_ID)
        assert result == expected
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["view_id"] == str(VIEW_ID)
        assert body["grantee_id"] == str(GRANTEE_ID)
        assert body["permissions"] == ["read"]

    async def test_revoke(self):
        expected = {"revoked": True}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.revoke(CAP_ID)
        assert result == expected
        path = self.mock_http.post.call_args.args[0]
        assert str(CAP_ID) in path
        assert "revoke" in path

    async def test_revoke_shared_view(self):
        expected = {"revoked": True}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.revoke_shared_view(AGENT_ID, CAP_ID)
        assert result == expected

    async def test_list_outbound_capabilities(self):
        expected = [{"capability_id": str(CAP_ID)}]
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.list_outbound_capabilities(AGENT_ID)
        assert result == expected
        params = self.mock_http.get.call_args.kwargs["params"]
        assert params["direction"] == "outbound"

    async def test_list_shared_views(self):
        expected = [{"name": "shared-view"}]
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.list_shared_views(AGENT_ID)
        assert result == expected

    async def test_recall_shared(self):
        expected = {"atoms": []}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.recall_shared(AGENT_ID, VIEW_ID, "query")
        assert result == expected

    async def test_recall_all_shared(self):
        expected = {"atoms": []}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.recall_all_shared(AGENT_ID, "query")
        assert result == expected
        path = self.mock_http.post.call_args.args[0]
        assert "shared_views/recall" in path

    async def test_recall_all_shared_with_from_agent(self):
        self.mock_http.post.return_value = _mock_response(200, {"atoms": []})

        await self.client.recall_all_shared(
            AGENT_ID, "query", from_agent="alice:org"
        )
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["from_agent"] == "alice:org"

    async def test_resolve_address(self):
        self.mock_http.get.return_value = _mock_response(
            200, {"agent_id": str(AGENT_ID)}
        )

        result = await self.client.resolve_address("alice:org")
        assert result == str(AGENT_ID)

    async def test_health(self):
        self.mock_http.get.return_value = _mock_response(200, {"status": "ok"})

        result = await self.client.health()
        assert result == {"status": "ok"}


# ── Operator-plane operation tests ───────────────────────────────────────────


class TestOperatorPlaneOps:
    def setup_method(self):
        self.client = MnemoClient(BASE_URL, operator_key=OPERATOR_KEY)
        self.mock_http = AsyncMock(spec=httpx.AsyncClient)
        self.client._operator_http = self.mock_http
        self.mock_http.is_closed = False

    async def test_register_agent(self):
        expected = {"id": str(uuid4()), "name": "my-agent"}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.register_agent("my-agent", persona="helpful")
        assert result == expected
        body = self.mock_http.post.call_args.kwargs["json"]
        assert body["name"] == "my-agent"
        assert body["persona"] == "helpful"

    async def test_me(self):
        expected = {"id": "op123", "name": "test-operator"}
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.me()
        assert result == expected
        assert "/v1/auth/me" in self.mock_http.get.call_args.args[0]

    async def test_find_agent_by_name(self):
        expected = [{"id": str(uuid4()), "name": "agent-1"}]
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.find_agent_by_name("agent-1")
        assert result == expected
        params = self.mock_http.get.call_args.kwargs["params"]
        assert params["name"] == "agent-1"

    async def test_depart(self):
        expected = {"status": "departing"}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.depart(AGENT_ID)
        assert result == expected
        assert "depart" in self.mock_http.post.call_args.args[0]

    async def test_inspect_shares(self):
        expected = {"inbound": [], "outbound": []}
        self.mock_http.get.return_value = _mock_response(200, expected)

        result = await self.client.inspect_shares()
        assert result == expected
        assert "shares" in self.mock_http.get.call_args.args[0]

    async def test_block_share(self):
        expected = {"blocked": True}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.block_share(str(CAP_ID))
        assert result == expected
        assert "block" in self.mock_http.post.call_args.args[0]

    async def test_unblock_share(self):
        expected = {"blocked": False}
        self.mock_http.post.return_value = _mock_response(200, expected)

        result = await self.client.unblock_share(str(CAP_ID))
        assert result == expected
        assert "unblock" in self.mock_http.post.call_args.args[0]


# ── get_agent uses either client ─────────────────────────────────────────────


class TestGetAgent:
    async def test_get_agent_prefers_agent_client(self):
        c = MnemoClient(BASE_URL, agent_key=AGENT_KEY, operator_key=OPERATOR_KEY)
        mock_agent = AsyncMock(spec=httpx.AsyncClient)
        mock_agent.is_closed = False
        c._agent_http = mock_agent
        expected = {"id": str(AGENT_ID), "name": "test"}
        mock_agent.get.return_value = _mock_response(200, expected)

        result = await c.get_agent(AGENT_ID)
        assert result == expected
        mock_agent.get.assert_awaited_once()

    async def test_get_agent_falls_back_to_operator(self):
        c = MnemoClient(BASE_URL, operator_key=OPERATOR_KEY)
        mock_op = AsyncMock(spec=httpx.AsyncClient)
        mock_op.is_closed = False
        c._operator_http = mock_op
        expected = {"id": str(AGENT_ID), "name": "test"}
        mock_op.get.return_value = _mock_response(200, expected)

        result = await c.get_agent(AGENT_ID)
        assert result == expected
        mock_op.get.assert_awaited_once()


# ── MnemoClientSync tests ───────────────────────────────────────────────────


class TestMnemoClientSync:
    def test_constructor_requires_agent_key(self):
        with pytest.raises(ValueError, match="agent_key is required"):
            MnemoClientSync(agent_key="", agent_id=str(AGENT_ID))

    def test_constructor_legacy_api_key(self):
        c = MnemoClientSync(agent_key="", agent_id=str(AGENT_ID), api_key="legacy")
        assert c._agent_key == "legacy"

    @patch("mnemo.client.MnemoClient", autospec=True)
    def test_remember_delegates(self, MockClient):
        mock_instance = AsyncMock()
        MockClient.return_value = mock_instance
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.remember.return_value = {"status": "queued", "store_id": "s1"}

        sync = MnemoClientSync(agent_key=AGENT_KEY, agent_id=str(AGENT_ID))
        result = sync.remember("test memory")
        assert result["store_id"] == "s1"

    @patch("mnemo.client.MnemoClient", autospec=True)
    def test_recall_delegates(self, MockClient):
        mock_instance = AsyncMock()
        MockClient.return_value = mock_instance
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.recall.return_value = {
            "atoms": [],
            "expanded_atoms": [],
            "total_retrieved": 0,
        }

        sync = MnemoClientSync(agent_key=AGENT_KEY, agent_id=str(AGENT_ID))
        result = sync.recall("search query")
        assert result["total_retrieved"] == 0

    @patch("mnemo.client.MnemoClient", autospec=True)
    def test_stats_delegates(self, MockClient):
        mock_instance = AsyncMock()
        MockClient.return_value = mock_instance
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.stats.return_value = {"total_atoms": 5, "active_atoms": 3}

        sync = MnemoClientSync(agent_key=AGENT_KEY, agent_id=str(AGENT_ID))
        result = sync.stats()
        assert result["total_atoms"] == 5
