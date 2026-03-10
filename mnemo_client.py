"""
Lightweight async Python client for the Mnemo API.

Usage:
    async with MnemoClient(api_key="your-key") as client:
        agent = await client.register_agent("my-agent",
            persona="python developer", domain_tags=["python"])

        result = await client.remember(
            agent_id=agent["id"],
            text="pandas.read_csv silently coerces mixed-type columns. "
                 "I discovered this processing client_data.csv. "
                 "From now on I should always specify dtype explicitly.",
            domain_tags=["python", "pandas"],
        )
        # result: RememberResult {atoms_created: 3, edges_created: 2, duplicates_merged: 0}

        results = await client.recall(
            agent_id=agent["id"],
            query="loading CSV files with pandas",
        )

        view = await client.create_view(
            agent_id=agent["id"],
            name="pandas-csv-handling",
            atom_filter={"atom_types": ["procedural"], "domain_tags": ["pandas"]},
        )
        skill = await client.export_skill(agent_id=agent["id"], view_id=view["id"])

        cap = await client.grant(
            agent_id=agent["id"],
            view_id=view["id"],
            grantee_id=other_agent["id"],
        )

Sync usage:
    client = MnemoClientSync(api_key="your-key", agent_id="your-uuid")
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
    atoms_created: int
    edges_created: int
    duplicates_merged: int


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
    def __init__(self, base_url: str = "https://api.mnemo.ai",
                 api_key: Optional[str] = None):
        if not api_key:
            raise ValueError(
                "api_key is required. Get yours at https://mnemo.ai"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

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

    # ── Primary Interface ──────────────────────────────────────────────────────

    async def remember(
        self,
        agent_id: UUID,
        text: str,
        domain_tags: Optional[list[str]] = None,
    ) -> RememberResult:
        """Store a memory. Just say what happened. Server decomposes and links."""
        resp = await self.http.post(
            f"/v1/agents/{_uid(agent_id)}/remember",
            json={"text": text, "domain_tags": domain_tags or []},
        )
        self._raise_for_status(resp)
        return resp.json()

    async def recall(
        self,
        agent_id: UUID,
        query: str,
        atom_types: Optional[list[str]] = None,
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
            "atom_types": atom_types,
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
        resp = await self.http.post(
            f"/v1/agents/{_uid(agent_id)}/recall",
            json=body,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Agent Management ───────────────────────────────────────────────────────

    async def register_agent(
        self,
        name: str,
        persona: Optional[str] = None,
        domain_tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        resp = await self.http.post(
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
        """GET /v1/auth/me — returns current agent info (requires api_key)."""
        resp = await self.http.get("/v1/auth/me")
        self._raise_for_status(resp)
        return resp.json()

    async def find_agent_by_name(self, name: str) -> list[dict]:
        """Return active agents with this exact name (empty list if none found)."""
        resp = await self.http.get("/v1/agents", params={"name": name})
        self._raise_for_status(resp)
        return resp.json()

    async def get_agent(self, agent_id: UUID) -> dict:
        resp = await self.http.get(f"/v1/agents/{_uid(agent_id)}")
        self._raise_for_status(resp)
        return resp.json()

    async def stats(self, agent_id: UUID) -> AgentStats:
        resp = await self.http.get(f"/v1/agents/{_uid(agent_id)}/stats")
        self._raise_for_status(resp)
        return resp.json()

    async def depart(self, agent_id: UUID) -> dict:
        """Initiate agent departure. Cascade-revokes all granted capabilities."""
        resp = await self.http.post(f"/v1/agents/{_uid(agent_id)}/depart")
        self._raise_for_status(resp)
        return resp.json()

    async def resolve_address(self, address: str) -> str:
        """Resolve agent address to agent UUID string. Raises MnemoNotFoundError if not found."""
        resp = await self.http.get(f"/v1/agents/resolve/{address}")
        self._raise_for_status(resp)
        return resp.json()["agent_id"]

    # ── Power-User Atom Operations ─────────────────────────────────────────────

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
        resp = await self.http.post(
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
        resp = await self.http.get(
            f"/v1/agents/{_uid(agent_id)}/atoms/{_uid(atom_id)}"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def delete_atom(self, agent_id: UUID, atom_id: UUID) -> None:
        resp = await self.http.delete(
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
        resp = await self.http.post(
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

    # ── View Operations ────────────────────────────────────────────────────────

    async def create_view(
        self,
        agent_id: UUID,
        name: str,
        atom_filter: dict,
        description: Optional[str] = None,
    ) -> dict:
        resp = await self.http.post(
            f"/v1/agents/{_uid(agent_id)}/views",
            json={"name": name, "description": description, "atom_filter": atom_filter},
        )
        self._raise_for_status(resp)
        return resp.json()

    async def list_views(self, agent_id: UUID) -> list[dict]:
        resp = await self.http.get(f"/v1/agents/{_uid(agent_id)}/views")
        self._raise_for_status(resp)
        return resp.json()

    async def export_skill(self, agent_id: UUID, view_id: UUID) -> dict:
        resp = await self.http.get(
            f"/v1/agents/{_uid(agent_id)}/views/{_uid(view_id)}/export_skill"
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Capability Operations ──────────────────────────────────────────────────

    async def grant(
        self,
        agent_id: UUID,
        view_id: UUID,
        grantee_id: UUID,
        permissions: Optional[list[str]] = None,
        expires_at: Optional[str] = None,
    ) -> dict:
        resp = await self.http.post(
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
        resp = await self.http.post(
            f"/v1/capabilities/{_uid(capability_id)}/revoke"
        )
        self._raise_for_status(resp)
        return resp.json()

    async def list_shared_views(self, agent_id: UUID) -> list[dict]:
        resp = await self.http.get(f"/v1/agents/{_uid(agent_id)}/shared_views")
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
        resp = await self.http.post(
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
        resp = await self.http.post(
            f"/v1/agents/{_uid(agent_id)}/shared_views/recall",
            json=body,
        )
        self._raise_for_status(resp)
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────────────

    async def health(self) -> dict:
        resp = await self.http.get("/v1/health")
        self._raise_for_status(resp)
        return resp.json()


# ── Sync wrapper ───────────────────────────────────────────────────────────────

class MnemoClientSync:
    """
    Synchronous wrapper around MnemoClient.

    Use this in non-async contexts (e.g. inside a synchronous agent loop,
    a WebSocket handler, or a script). Handles the case where an event
    loop is already running by offloading to a worker thread.

    Example:
        client = MnemoClientSync(api_key="your-key", agent_id="your-uuid")
        client.remember("pandas.read_csv coerces mixed types")
        results = client.recall("loading CSV files")
    """

    def __init__(self, api_key: str, agent_id: str,
                 base_url: str = "https://api.mnemo.ai"):
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
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
                 domain_tags: list[str] | None = None) -> RememberResult:
        async def _():
            async with MnemoClient(self._base_url, self._api_key) as c:
                return await c.remember(self._agent_id, text, domain_tags)
        return self._run(_())

    def recall(self, query: str, **kwargs) -> RecallResult:
        async def _():
            async with MnemoClient(self._base_url, self._api_key) as c:
                return await c.recall(self._agent_id, query, **kwargs)
        return self._run(_())

    def stats(self) -> AgentStats:
        async def _():
            async with MnemoClient(self._base_url, self._api_key) as c:
                return await c.stats(self._agent_id)
        return self._run(_())
