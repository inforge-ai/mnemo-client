"""
Lightweight async Python client for the Mnemo API.

Usage (agent operations — remember, recall, share, stats):
    async with MnemoClient(agent_key="mnemo_ag_...") as client:
        result = await client.remember(agent_id=..., text="...")
        results = await client.recall(agent_id=..., query="...")

Usage (operator operations — register agent, inspect shares, block):
    async with MnemoClient(operator_key="mnemo_op_...") as client:
        agent = await client.register_agent("my-agent")
        shares = await client.inspect_shares()

Usage (both planes in one client):
    async with MnemoClient(
        agent_key="mnemo_ag_...",
        operator_key="mnemo_op_...",
    ) as client:
        agent = await client.register_agent("my-agent")
        await client.remember(agent_id=agent["id"], text="...")

Sync usage:
    client = MnemoClientSync(agent_key="mnemo_ag_...", agent_id="your-uuid")
    client.remember("pandas.read_csv coerces mixed types")
    results = client.recall("loading CSV files")
"""

import asyncio
import concurrent.futures
from typing import Optional, TypedDict
from uuid import UUID

import httpx


# ── Exceptions ─────────────────────────────────────────────────────────────────

class MnemoError(Exception):
    """Base exception for all Mnemo client errors."""
    pass


class MnemoAuthError(MnemoError):
    """Raised on 401 or 403 responses."""
    pass


class MnemoNotFoundError(MnemoError):
    """Raised on 404 responses."""
    pass


class MnemoServerError(MnemoError):
    """Raised on 5xx responses."""
    pass


# ── Typed response shapes ──────────────────────────────────────────────────────

class RememberResult(TypedDict):
    status: str
    store_id: str


class Atom(TypedDict):
    id: str
    atom_type: str
    text_content: str
    confidence_expected: float
    confidence_effective: float
    domain_tags: list[str]
    source_type: str
    created_at: str
    access_count: int


class RecallResult(TypedDict):
    atoms: list[Atom]
    expanded_atoms: list[Atom]
    total_retrieved: int


class AgentStats(TypedDict):
    agent_id: str
    total_atoms: int
    active_atoms: int
    atoms_by_type: dict
    total_edges: int
    avg_effective_confidence: float
    active_views: int
    granted_capabilities: int
    received_capabilities: int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _uid(v) -> str:
    """Normalise UUID or string to plain string for HTTP."""
    return str(v)


# ── Async client ───────────────────────────────────────────────────────────────

class MnemoClient:
    def __init__(
        self,
        base_url: str = "https://api.mnemo-ai.com",
        agent_key: Optional[str] = None,
        operator_key: Optional[str] = None,
        # Legacy support: api_key maps to operator_key
        api_key: Optional[str] = None,
    ):
        if api_key and not operator_key:
            operator_key = api_key
        if not agent_key and not operator_key:
            raise ValueError(
                "At least one of agent_key or operator_key is required. "
                "Use agent_key for data-plane ops (remember, recall, share). "
                "Use operator_key for management ops (register agent, inspect shares)."
            )
        self.base_url = base_url.rstrip("/")
        self.agent_key = agent_key
        self.operator_key = operator_key
        self._agent_http: httpx.AsyncClient | None = None
        self._operator_http: httpx.AsyncClient | None = None

    @property
    def _agent_client(self) -> httpx.AsyncClient:
        """HTTP client with X-Agent-Key header for data-plane operations."""
        if not self.agent_key:
            raise MnemoAuthError(
                "This operation requires an agent key. "
                "Pass agent_key= to MnemoClient."
            )
        if self._agent_http is None or self._agent_http.is_closed:
            self._agent_http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"X-Agent-Key": self.agent_key},
            )
        return self._agent_http

    @property
    def _operator_client(self) -> httpx.AsyncClient:
        """HTTP client with X-Operator-Key header for management-plane operations."""
        if not self.operator_key:
            raise MnemoAuthError(
                "This operation requires an operator key. "
                "Pass operator_key= to MnemoClient."
            )
        if self._operator_http is None or self._operator_http.is_closed:
            self._operator_http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"X-Operator-Key": self.operator_key},
            )
        return self._operator_http

    @property
    def http(self) -> httpx.AsyncClient:
        """Legacy property — returns agent client if available, otherwise operator."""
        if self.agent_key:
            return self._agent_client
        return self._operator_client

    async def close(self):
        if self._agent_http and not self._agent_http.is_closed:
            await self._agent_http.aclose()
        if self._operator_http and not self._operator_http.is_closed:
            await self._operator_http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise MnemoAuthError("Invalid or missing API key")
        if resp.status_code == 403:
            raise MnemoAuthError(f"Permission denied: {resp.text}")
        if resp.status_code == 404:
            raise MnemoNotFoundError(resp.text)
        if resp.status_code >= 500:
            raise MnemoServerError(
                f"Mnemo server error {resp.status_code}: {resp.text}"
            )
        resp.raise_for_status()

    # ── Primary Interface (agent key) ─────────────────────────────────────────

    async def remember(
        self,
        agent_id: UUID,
        text: str,
        domain_tags: Optional[list[str]] = None,
        remembered_on: Optional[str] = None,
    ) -> RememberResult:
        """Store a memory. Just say what happened. Server decomposes and links."""
        body: dict = {"text": text, "domain_tags": domain_tags or []}
        if remembered_on is not None:
            body["remembered_on"] = remembered_on
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/remember",
            json=body,
        )
        self._raise_for_status(resp)
        return resp.json()

    async def recall(
        self,
        agent_id: UUID,
        query: str,
        domain_tags: Optional[list[str]] = None,
        min_confidence: float = 0.1,
        min_similarity: float = 0.2,
        max_results: int = 10,
        expand_graph: bool = True,
        expansion_depth: int = 2,
        include_superseded: bool = False,
        similarity_drop_threshold: Optional[float] = None,
        verbosity: Optional[str] = None,
        max_total_tokens: Optional[int] = None,
    ) -> RecallResult:
        """Retrieve relevant memories by semantic search."""
        body: dict = {
            "query": query,
            "domain_tags": domain_tags,
            "min_confidence": min_confidence,
            "min_similarity": min_similarity,
            "max_results": max_results,
            "expand_graph": expand_graph,
            "expansion_depth": expansion_depth,
            "include_superseded": include_superseded,
        }
        if similarity_drop_threshold is not None:
            body["similarity_drop_threshold"] = similarity_drop_threshold
        if verbosity is not None:
            body["verbosity"] = verbosity
        if max_total_tokens is not None:
            body["max_total_tokens"] = max_total_tokens
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/recall",
            json=body,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Agent Management (operator key) ───────────────────────────────────────

    async def register_agent(
        self,
        name: str,
        persona: Optional[str] = None,
        domain_tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        resp = await self._operator_client.post(
            "/v1/agents",
            json={
                "name": name,
                "persona": persona,
                "domain_tags": domain_tags or [],
                "metadata": metadata or {},
            },
        )
        self._raise_for_status(resp)
        return resp.json()

    async def me(self) -> dict:
        """GET /v1/auth/me — returns current operator info."""
        resp = await self._operator_client.get("/v1/auth/me")
        self._raise_for_status(resp)
        return resp.json()

    async def find_agent_by_name(self, name: str) -> list[dict]:
        """Return active agents with this exact name (empty list if none found)."""
        resp = await self._operator_client.get("/v1/agents", params={"name": name})
        self._raise_for_status(resp)
        return resp.json()

    async def get_agent(self, agent_id: UUID) -> dict:
        # get_agent is a read-only lookup, use whichever client is available
        client = self._agent_client if self.agent_key else self._operator_client
        resp = await client.get(f"/v1/agents/{_uid(agent_id)}")
        self._raise_for_status(resp)
        return resp.json()

    async def stats(self, agent_id: UUID) -> AgentStats:
        resp = await self._agent_client.get(f"/v1/agents/{_uid(agent_id)}/stats")
        self._raise_for_status(resp)
        return resp.json()

    async def depart(self, agent_id: UUID) -> dict:
        """Initiate agent departure (admin only in RBAC-Lite)."""
        resp = await self._operator_client.post(f"/v1/agents/{_uid(agent_id)}/depart")
        self._raise_for_status(resp)
        return resp.json()

    async def resolve_address(self, address: str) -> str:
        """Resolve agent address to agent UUID string."""
        client = self._agent_client if self.agent_key else self._operator_client
        resp = await client.get(f"/v1/agents/resolve/{address}")
        self._raise_for_status(resp)
        return resp.json()["agent_id"]

    # ── Power-User Atom Operations (agent key) ────────────────────────────────

    async def store_atom(
        self,
        agent_id: UUID,
        atom_type: str,
        text_content: str,
        structured: Optional[dict] = None,
        confidence: Optional[str] = None,
        source_type: str = "direct_experience",
        domain_tags: Optional[list[str]] = None,
    ) -> dict:
        """Explicit atom creation (power-user interface)."""
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/atoms",
            json={
                "atom_type": atom_type,
                "text_content": text_content,
                "structured": structured or {},
                "confidence": confidence,
                "source_type": source_type,
                "domain_tags": domain_tags or [],
            },
        )
        self._raise_for_status(resp)
        return resp.json()

    async def get_atom(self, agent_id: UUID, atom_id: UUID) -> dict:
        resp = await self._agent_client.get(
            f"/v1/agents/{_uid(agent_id)}/atoms/{_uid(atom_id)}"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def delete_atom(self, agent_id: UUID, atom_id: UUID) -> None:
        resp = await self._agent_client.delete(
            f"/v1/agents/{_uid(agent_id)}/atoms/{_uid(atom_id)}"
        )
        self._raise_for_status(resp)

    async def link(
        self,
        agent_id: UUID,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        weight: float = 1.0,
    ) -> dict:
        """Create a typed edge between two atoms."""
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/atoms/link",
            json={
                "source_id": _uid(source_id),
                "target_id": _uid(target_id),
                "edge_type": edge_type,
                "weight": weight,
            },
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── View Operations (agent key) ───────────────────────────────────────────

    async def create_view(
        self,
        agent_id: UUID,
        name: str,
        atom_filter: dict,
        description: Optional[str] = None,
    ) -> dict:
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/views",
            json={"name": name, "description": description, "atom_filter": atom_filter},
        )
        self._raise_for_status(resp)
        return resp.json()

    async def list_views(self, agent_id: UUID) -> list[dict]:
        resp = await self._agent_client.get(f"/v1/agents/{_uid(agent_id)}/views")
        self._raise_for_status(resp)
        return resp.json()

    async def export_skill(self, agent_id: UUID, view_id: UUID) -> dict:
        resp = await self._agent_client.get(
            f"/v1/agents/{_uid(agent_id)}/views/{_uid(view_id)}/export_skill"
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Capability Operations (agent key) ─────────────────────────────────────

    async def grant(
        self,
        agent_id: UUID,
        view_id: UUID,
        grantee_id: UUID,
        permissions: Optional[list[str]] = None,
        expires_at: Optional[str] = None,
    ) -> dict:
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/grant",
            json={
                "view_id": _uid(view_id),
                "grantee_id": _uid(grantee_id),
                "permissions": permissions or ["read"],
                "expires_at": expires_at,
            },
        )
        self._raise_for_status(resp)
        return resp.json()

    async def revoke(self, capability_id: UUID) -> dict:
        resp = await self._agent_client.post(
            f"/v1/capabilities/{_uid(capability_id)}/revoke"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def revoke_shared_view(
        self, agent_id: UUID, capability_id: UUID
    ) -> dict:
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/capabilities/{_uid(capability_id)}/revoke"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def list_outbound_capabilities(self, agent_id: UUID) -> list[dict]:
        resp = await self._agent_client.get(
            f"/v1/agents/{_uid(agent_id)}/capabilities",
            params={"direction": "outbound"},
        )
        self._raise_for_status(resp)
        return resp.json()

    async def list_shared_views(self, agent_id: UUID) -> list[dict]:
        resp = await self._agent_client.get(f"/v1/agents/{_uid(agent_id)}/shared_views")
        self._raise_for_status(resp)
        return resp.json()

    async def recall_shared(
        self,
        agent_id: UUID,
        view_id: UUID,
        query: str,
        min_confidence: float = 0.1,
        max_results: int = 10,
        expand_graph: bool = True,
        expansion_depth: int = 2,
    ) -> dict:
        """Recall through a shared view (scope-bounded to snapshot atoms)."""
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/shared_views/{_uid(view_id)}/recall",
            json={
                "query": query,
                "min_confidence": min_confidence,
                "max_results": max_results,
                "expand_graph": expand_graph,
                "expansion_depth": expansion_depth,
            },
        )
        self._raise_for_status(resp)
        return resp.json()

    async def recall_all_shared(
        self,
        agent_id: UUID,
        query: str,
        from_agent: str | None = None,
        min_similarity: float = 0.15,
        max_results: int = 5,
        verbosity: str = "summary",
        max_total_tokens: int | None = None,
    ) -> dict:
        """Recall across all shared views (cross-view endpoint)."""
        body: dict = {
            "query": query,
            "min_similarity": min_similarity,
            "max_results": max_results,
            "verbosity": verbosity,
        }
        if from_agent:
            body["from_agent"] = from_agent
        if max_total_tokens is not None:
            body["max_total_tokens"] = max_total_tokens
        resp = await self._agent_client.post(
            f"/v1/agents/{_uid(agent_id)}/shared_views/recall",
            json=body,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Operator Share Management (operator key) ──────────────────────────────

    async def inspect_shares(self) -> dict:
        """List all inbound and outbound shares for this operator's agents."""
        resp = await self._operator_client.get("/v1/operators/me/shares")
        self._raise_for_status(resp)
        return resp.json()

    async def block_share(self, capability_id: str) -> dict:
        """Block an inbound share to one of this operator's agents."""
        resp = await self._operator_client.post(
            f"/v1/shares/{capability_id}/block"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def unblock_share(self, capability_id: str) -> dict:
        """Unblock a previously blocked inbound share."""
        resp = await self._operator_client.post(
            f"/v1/shares/{capability_id}/unblock"
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────────────

    async def health(self) -> dict:
        client = self._agent_client if self.agent_key else self._operator_client
        resp = await client.get("/v1/health")
        self._raise_for_status(resp)
        return resp.json()


# ── Sync wrapper ───────────────────────────────────────────────────────────────

class MnemoClientSync:
    """
    Synchronous wrapper around MnemoClient.

    Use this in non-async contexts (e.g. inside a synchronous agent loop,
    a WebSocket handler, or a script).
    """

    def __init__(self, agent_key: str, agent_id: str,
                 base_url: str = "https://api.mnemo-ai.com",
                 # Legacy support
                 api_key: Optional[str] = None):
        key = agent_key or api_key
        if not key:
            raise ValueError("agent_key is required")
        self._agent_key = key
        self._agent_id = agent_id
        self._base_url = base_url

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    def remember(self, text: str,
                 domain_tags: list[str] | None = None,
                 remembered_on: str | None = None) -> RememberResult:
        async def _():
            async with MnemoClient(self._base_url, agent_key=self._agent_key) as c:
                return await c.remember(
                    self._agent_id, text, domain_tags,
                    remembered_on=remembered_on,
                )
        return self._run(_())

    def recall(self, query: str, **kwargs) -> RecallResult:
        async def _():
            async with MnemoClient(self._base_url, agent_key=self._agent_key) as c:
                return await c.recall(self._agent_id, query, **kwargs)
        return self._run(_())

    def stats(self) -> AgentStats:
        async def _():
            async with MnemoClient(self._base_url, agent_key=self._agent_key) as c:
                return await c.stats(self._agent_id)
        return self._run(_())
