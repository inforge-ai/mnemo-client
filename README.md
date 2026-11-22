# mnemo-ai

Persistent memory for AI agents — async Python client library + MCP server.

## Installation

### Client library only

```bash
pip install mnemo-ai
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add mnemo-ai
```

### With MCP server (for Claude Desktop, etc.)

```bash
pip install "mnemo-ai[mcp]"
```

Or with uv:

```bash
uv add "mnemo-ai[mcp]"
```

## Quick Start

### Async (recommended)

The client uses RBAC headers. Use `agent_key` for data-plane operations (remember, recall, stats, views, capabilities) and `operator_key` for management operations (register agent, list agents, inspect shares):

```python
from mnemo.client import MnemoClient

# Agent operations only (most common)
async with MnemoClient(agent_key="mnemo_ag_...") as client:
    result = await client.remember(
        agent_id="your-agent-uuid",
        text="pandas.read_csv silently coerces mixed-type columns. "
             "I discovered this processing client_data.csv. "
             "From now on I should always specify dtype explicitly.",
        domain_tags=["python", "pandas"],
    )

    results = await client.recall(
        agent_id="your-agent-uuid",
        query="loading CSV files with pandas",
    )

# Operator operations only
async with MnemoClient(operator_key="mnemo_op_...") as client:
    agent = await client.register_agent(
        "my-agent",
        persona="python developer",
        domain_tags=["python"],
    )

# Both planes in one client
async with MnemoClient(
    agent_key="mnemo_ag_...",
    operator_key="mnemo_op_...",
) as client:
    agent = await client.register_agent("my-agent")
    await client.remember(agent_id=agent["id"], text="...")
```

### Sync

For non-async contexts (scripts, sync agent loops, WebSocket handlers):

```python
from mnemo.client import MnemoClientSync

client = MnemoClientSync(agent_key="mnemo_ag_...", agent_id="your-uuid")

client.remember("pandas.read_csv coerces mixed types")
results = client.recall("loading CSV files")
stats = client.stats()
```

`MnemoClientSync` safely handles the case where an event loop is already running by offloading work to a worker thread.

## MCP Server for Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "mnemo": {
      "command": "uvx",
      "args": ["mnemo-ai[mcp]"],
      "env": {
        "MNEMO_AGENT_KEY": "mnemo_ag_...",
        "MNEMO_BASE_URL": "https://your-mnemo-server.com"
      }
    }
  }
}
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `MNEMO_AGENT_KEY` | Yes | Agent key for data-plane authentication (sends `X-Agent-Key` header) |
| `MNEMO_BASE_URL` | Yes | URL of the Mnemo server |
| `MNEMO_DEFAULT_AGENT_ID` | No | Default agent (UUID, name or address). If unset, tools use the agent the key belongs to (`GET /v1/agents/me`). |
| `MNEMO_MCP_TRANSPORT` | No | Transport type: `stdio` (default), `sse`, or `streamable-http` |
| `MNEMO_MCP_HOST` | No | Bind address for HTTP/SSE transports (default `127.0.0.1`) |
| `MNEMO_MCP_PORT` | No | Port for HTTP/SSE transports (default `8001`) |

> **Legacy support:** `MNEMO_API_KEY` is still accepted as a fallback if `MNEMO_AGENT_KEY` is not set.

### MCP tools

The server exposes 10 tools:

| Tool | What it does |
|---|---|
| `mnemo_remember` | Store a memory and get back exactly what was stored: one line per atom, with its id, type, date and any flags. Optional `speaker`, `addressee`, `channel`, `origin_class`, `remembered_on`. Abbreviations and codes are stored exactly as written; spell them out in the text if the meaning matters. If the server cannot decompose the text, it is kept and the reply says how to retry. |
| `mnemo_recall` | Search memories. `since` / `until` (ISO dates, on when the memory happened) and `atom_types` filter on the server. Default `verbosity="full"` returns each memory complete up to `max_chars_per_atom` (600); `"summary"` returns first sentences. |
| `mnemo_recent` | Newest memories without searching; `order_by="stored"` (default) or `"event"`. |
| `mnemo_forget` | Delete one memory by atom id (audited as deleted by the agent). |
| `mnemo_store_status` | Status and atoms of a store; `retry=True` re-runs a failed store. |
| `mnemo_stats` | Memory statistics. |
| `mnemo_share`, `mnemo_list_shared`, `mnemo_recall_shared`, `mnemo_revoke_share` | Sharing (unchanged). |

`agent_id` is optional on every tool and accepts a UUID, an agent name (`clio`) or an address (`clio:tom.inforge`). Every line of remember, recall and recent output starts with the full atom id, so `mnemo_forget` can act on it.

Requires a mnemo server with v2 Phase 1 (`/v1/agents/me`, synchronous remember). Against an older server, `mnemo_remember` reports "queued".

`mnemo_list_shared` displays a `[trusted]` or `[UNTRUSTED]` label per view based on the server's trust auth layer. Trust is managed by operators via the server CLI, not through MCP tools.

### Running manually

```bash
# Via console script
mnemo-mcp

# Via Python module
python -m mnemo.mcp
```

## API Reference

### `MnemoClient(base_url, agent_key=None, operator_key=None, api_key=None)`

Async client. Use as an async context manager or call `.close()` manually.

- `agent_key` — sends `X-Agent-Key` header for data-plane operations (remember, recall, stats, views, capabilities)
- `operator_key` — sends `X-Operator-Key` header for management operations (register agent, list agents, inspect shares, block/unblock)
- `api_key` — legacy alias for `operator_key`

At least one of `agent_key` or `operator_key` is required.

#### Memory

| Method | Description |
|---|---|
| `remember(agent_id, text, domain_tags=None, remembered_on=None, speaker=None, addressee=None, channel=None, origin_class=None, wait=True)` | Store a memory. The server decomposes it into typed atoms and links them, and (with `wait=True`) returns them. `remembered_on` accepts an ISO-8601 datetime to anchor relative time references. Raises `MnemoStoreFailedError` if decomposition fails (the text is kept). Returns `RememberResult`. 120 s timeout. |
| `store_status(store_id)` / `retry_store(store_id)` | Status and atoms of a store; re-run a failed store. |
| `recent(agent_id, limit=20, since=None, until=None, atom_types=None, domain_tags=None, order_by="stored")` | Newest memories without searching. |
| `whoami()` | The agent the key belongs to (`id`, `name`, `address`, `status`). |
| `recall(agent_id, query, ..., since=None, until=None, atom_types=None, max_content_chars=None)` | Semantic search over stored memories. Returns `RecallResult`. |

#### Agent Management

| Method | Description |
|---|---|
| `register_agent(name, persona=None, domain_tags=None, metadata=None)` | Create a new agent. |
| `me()` | Return info for the authenticated agent. |
| `find_agent_by_name(name)` | Find active agents by exact name. |
| `get_agent(agent_id)` | Get agent by ID. |
| `stats(agent_id)` | Return memory statistics for an agent. |
| `depart(agent_id)` | Deactivate an agent and cascade-revoke all granted capabilities. |

#### Atoms (Power-User)

| Method | Description |
|---|---|
| `store_atom(agent_id, atom_type, text_content, ...)` | Explicitly create a single memory atom. |
| `get_atom(agent_id, atom_id)` | Fetch a single atom. |
| `delete_atom(agent_id, atom_id)` | Delete an atom. |
| `link(agent_id, source_id, target_id, edge_type, weight=1.0)` | Create a typed edge between two atoms. |

#### Views & Skills

| Method | Description |
|---|---|
| `create_view(agent_id, name, atom_filter, description=None)` | Create a filtered view over an agent's atoms. |
| `list_views(agent_id)` | List all views for an agent. |
| `export_skill(agent_id, view_id)` | Export a view as a portable skill bundle. |

#### Capability Sharing

| Method | Description |
|---|---|
| `grant(agent_id, view_id, grantee_id, permissions=None, expires_at=None)` | Grant another agent access to a view. |
| `revoke(capability_id)` | Revoke a previously granted capability. |
| `revoke_shared_view(agent_id, capability_id)` | Revoke a capability scoped to an agent. |
| `list_outbound_capabilities(agent_id)` | List capabilities this agent has granted to others. |
| `list_shared_views(agent_id)` | List views shared with this agent. Returns `trusted` boolean per view. |
| `recall_shared(agent_id, view_id, query, ...)` | Recall memories scoped to a single shared view. |
| `recall_all_shared(agent_id, query, ...)` | Recall across all shared views (cross-view search). |
| `resolve_address(address)` | Resolve an agent address (e.g. `"name:org"`) to a UUID. |

#### Operator Share Management

| Method | Description |
|---|---|
| `inspect_shares()` | List all inbound and outbound shares for this operator's agents. |
| `block_share(capability_id)` | Block an inbound share to one of this operator's agents. |
| `unblock_share(capability_id)` | Unblock a previously blocked inbound share. |

#### Other

| Method | Description |
|---|---|
| `health()` | Check API health. |

### `MnemoClientSync(agent_key, agent_id, base_url)`

Synchronous wrapper for agent data-plane operations. Exposes `remember`, `recall`, and `stats` with the same signatures as the async client. Legacy `api_key` parameter is accepted as a fallback for `agent_key`.

## Response Types

- **`RememberResult`** — `status` (`complete` or `queued`), `store_id`, `source_id`, `atoms_created`, `duplicates_merged`, `atoms` (each: `id`, `atom_type`, `text_content`, `remembered_on`, `source_quote`, `merged_into_existing`, `flags`), `warnings`
- **`RecallResult`** — `atoms`, `expanded_atoms`, `total_retrieved`
- **`AgentStats`** — `agent_id`, `total_atoms`, `active_atoms`, `atoms_by_type`, `total_edges`, `avg_effective_confidence`, `active_views`, `granted_capabilities`, `received_capabilities`

## Exceptions

| Exception | Raised when |
|---|---|
| `MnemoAuthError` | 401 or 403 response |
| `MnemoNotFoundError` | 404 response |
| `MnemoServerError` | 5xx response |
| `MnemoStoreFailedError` | `remember` / `retry_store` failed server-side; has `store_id`, `source_id`, `error` (subclass of `MnemoServerError`) |
| `MnemoError` | Base class for all Mnemo errors |

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
