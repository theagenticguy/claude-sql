# Security Policy

## Supported versions

claude-sql ships from a single `main` line. Security fixes land on the latest
released minor and are published to PyPI. Older releases receive no backports;
upgrade to the latest version.

| Version | Supported |
| ------- | --------- |
| latest `1.x` | yes |
| < latest `1.x` | no: upgrade |

## What claude-sql touches

claude-sql reads your local Claude Code transcripts under `~/.claude/projects/`
(or an S3 corpus you point it at) into an in-memory DuckDB connection. It writes
derived artifacts (parquet shards, a LanceDB vector store, a SQLite checkpoint)
under `~/.claude-sql/`. It calls Amazon Bedrock for embeddings and LLM
analytics, authenticating through the standard AWS credential chain. It never
transmits transcript content anywhere except the Bedrock endpoints you
configure, and it holds no long-lived secrets of its own.

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub Security Advisories on
the repository (`Security` tab → `Report a vulnerability`), or to the maintainer
listed in `pyproject.toml`. Please do not open a public issue for a security
report.

Include the version, a description, and a reproduction if you have one. Expect
an acknowledgement within a few business days. Accepted reports are fixed on
`main` and released promptly; declined reports get a written rationale.
