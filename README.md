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

```python
from mnemo.client import MnemoClient

async with MnemoClient(api_key="your-key") as client:
    # Register an agent
    agent = await client.register_agent(
        "my-agent",
        persona="python developer",
        domain_tags=["python"],
    )

    # Store a memory
    result = await client.remember(
        agent_id=agent["id"],
        text="pandas.read_csv silently coerces mixed-type columns. "
             "I discovered this processing client_data.csv. "
             "From now on I should always specify dtype explicitly.",
        domain_tags=["python", "pandas"],
    )

    # Retrieve relevant memories
    results = await client.recall(
        agent_id=agent["id"],
        query="loading CSV files with pandas",
    )
```

### Sync

For non-async contexts (scripts, sync agent loops, WebSocket handlers):

```python
from mnemo.client import MnemoClientSync

client = MnemoClientSync(api_key="your-key", agent_id="your-uuid")

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
        "MNEMO_API_KEY": "your-api-key",
        "MNEMO_BASE_URL": "https://your-mnemo-server.com",
        "MNEMO_DEFAULT_AGENT_ID": "your-agent-uuid"
      }
    }
  }
}
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `MNEMO_API_KEY` | Yes | API key for authentication |
| `MNEMO_BASE_URL` | Yes | URL of the Mnemo server |
| `MNEMO_DEFAULT_AGENT_ID` | No | Default agent UUID (so tools work without specifying agent_id) |
| `MNEMO_MCP_TRANSPORT` | No | Transport type: `stdio` (default), `sse`, or `streamable-http` |

### MCP tools

The server exposes 7 tools: `mnemo_remember`, `mnemo_recall`, `mnemo_stats`, `mnemo_share`, `mnemo_list_shared`, `mnemo_recall_shared`, `mnemo_revoke_share`.

`mnemo_list_shared` displays a `[trusted]` or `[UNTRUSTED]` label per view based on the server's trust auth layer. Trust is managed by operators via the server CLI, not through MCP tools.

### Running manually

```bash
# Via console script
mnemo-mcp

# Via Python module
python -m mnemo.mcp
```

## API Reference

### `MnemoClient(api_key, base_url)`

Async client. Use as an async context manager or call `.close()` manually.

#### Memory

| Method | Description |
|---|---|
| `remember(agent_id, text, domain_tags=None)` | Store a memory. The server decomposes it into typed atoms and links them. Returns `RememberResult`. |
| `recall(agent_id, query, ...)` | Semantic search over stored memories. Returns `RecallResult`. |

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
| `list_shared_views(agent_id)` | List views shared with this agent. Returns `trusted` boolean per view. |
| `recall_shared(agent_id, view_id, query, ...)` | Recall memories scoped to a shared view. |

#### Other

| Method | Description |
|---|---|
| `health()` | Check API health. |

### `MnemoClientSync(api_key, agent_id, base_url)`

Synchronous wrapper. Exposes `remember`, `recall`, and `stats` with the same signatures as the async client.

## Response Types

- **`RememberResult`** — `atoms_created`, `edges_created`, `duplicates_merged`
- **`RecallResult`** — `atoms`, `expanded_atoms`, `total_retrieved`
- **`AgentStats`** — `total_atoms`, `active_atoms`, `atoms_by_type`, `total_edges`, `avg_effective_confidence`, `active_views`, `granted_capabilities`, `received_capabilities`

## Exceptions

| Exception | Raised when |
|---|---|
| `MnemoAuthError` | 401 or 403 response |
| `MnemoNotFoundError` | 404 response |
| `MnemoServerError` | 5xx response |
| `MnemoError` | Base class for all Mnemo errors |

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
