# claude-sql · Sequences

Diagram-only companion to [`behavior/processes.md`](../../behavior/processes.md)
and [`architecture/data-flow.md`](../../architecture/data-flow.md). One
`sequenceDiagram` per top process, showing the outbound call order across
participants. Every participant maps to a real module in the current
`packages/*/src/claude_sql/` layout.

## query

Read-only SQL over the DuckDB catalog — no Bedrock, no cost. Entry at
`packages/app/src/claude_sql/app/cli.py:728`; body at `cli.py:786-803`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as cli.query
    participant DuckDB as DuckDB+views
    participant Out as output

    User->>CLI: query cmd
    CLI->>CLI: resolve cfg
    CLI->>DuckDB: open+register
    DuckDB-->>CLI: connection
    CLI->>DuckDB: execute sql
    DuckDB-->>CLI: DataFrame
    CLI->>Out: emit_dataframe
    Out-->>User: table or JSON
    CLI->>DuckDB: close
```

## embed

Embeds unembedded messages via Cohere Embed v4 on Bedrock and appends
FLOAT[1024] vectors to LanceDB. Entry at `cli.py:1559`; body at
`cli.py:1603-1628`; worker `run_backfill` at
`packages/analytics/src/claude_sql/analytics/embed_worker.py:365`;
`discover_unembedded` at `embed_worker.py:100`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as cli.embed
    participant Worker as embed_worker
    participant DuckDB as DuckDB+views
    participant Bedrock as Bedrock Cohere
    participant Lance as lance_store

    User->>CLI: embed cmd
    CLI->>DuckDB: register views
    CLI->>Worker: run_backfill
    Worker->>Lance: get_uuids
    Lance-->>Worker: embedded set
    Worker->>DuckDB: scan pending
    DuckDB-->>Worker: uuid text
    Worker->>Bedrock: invoke_model
    Bedrock-->>Worker: vectors
    Worker->>Lance: add_chunk
    Worker-->>CLI: rows written
    CLI-->>User: result JSON
```

## classify

Classifies sessions with Sonnet 4.6 structured output and writes parquet
shards with session-level checkpointing. Entry at `cli.py:1730`; body at
`cli.py:1778-1793`; worker `_classify_sessions_async` at
`packages/analytics/src/claude_sql/analytics/classify_worker.py:45`;
`classify_one` at `packages/core/src/claude_sql/core/llm_shared.py:563`.

```mermaid
sequenceDiagram
    participant User
    participant CLI as cli.classify
    participant Worker as classify_worker
    participant Ckpt as checkpointer
    participant LLM as llm_shared
    participant Bedrock as Bedrock Sonnet
    participant Shards as parquet_shards

    User->>CLI: classify cmd
    CLI->>Worker: classify_run
    Worker->>Ckpt: filter
    Ckpt-->>Worker: pending sids
    Worker->>LLM: classify_one
    LLM->>Bedrock: invoke_model
    Bedrock-->>LLM: json output
    LLM-->>Worker: parsed dict
    Worker->>Shards: write_part
    Worker->>Ckpt: mark_completed
    Worker-->>CLI: rows written
    CLI-->>User: result JSON
```

## See also

- [claude-sql · Contract map](../../insights/contract-map.md) — 5 shared source files
- [claude-sql · Processes](../../behavior/processes.md) — 5 shared source files
- [claude-sql · Data flow](../../architecture/data-flow.md) — 4 shared source files
- [claude-sql · Debugging guide](../../insights/debugging-guide.md) — 3 shared source files
- [claude-sql · Module map](../../architecture/module-map.md) — 3 shared source files
