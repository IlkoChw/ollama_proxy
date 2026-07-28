# ollama_proxy

FastAPI proxy for [ollama.com](https://ollama.com) with **API-key rotation**
across multiple accounts. Each request goes through the next available
key; when upstream fails, the proxy rotates.

See [`AGENTS.md`](./AGENTS.md) for the full architecture overview.

## Features

- Async end-to-end (FastAPI + httpx + SQLAlchemy 2.0 async + aiosqlite).
- SQLite-backed key store — no external DB required.
- Round-robin rotation with `active` / `depleted` / `disabled` lifecycle
  and automatic cooldown on `429`.
- Admin bearer-token auth — `/admin/*` requires `ADMIN_TOKEN` (fail-closed).
- CRUD + probe endpoints (`/admin/keys/...`).
- In-memory model-list cache with TTL.
- OpenAI-compatible `/v1/chat/completions` and `/v1/models`.

## Dashboard (web 1.0 on purpose)

The optional dashboard (`ENABLE_DASHBOARD=1`) is intentionally styled
in **web 1.0 / Win9x** fashion — a deliberate aesthetic choice, not a
throwback. The chrome stays out of the way; the data is the page.

## Quick start

### Docker

```bash
cp .env.example .env
# PROBE_MODEL is REQUIRED — pick a model every upstream account can access.
docker compose up --build
```

SQLite persists in `./data/`. Swagger UI: <http://localhost:8000/docs>
(only when `ENABLE_DOCS=1`).

### Production behind Caddy

`docker-compose.prod.yml` is self-contained (does not inherit from
`docker-compose.yml`). Caddy terminates TLS on `:443`; `:8000` is **not**
published. Set `CADDY_DOMAIN` / `CADDY_EMAIL` / `ADMIN_TOKEN` in `.env`,
then `docker compose -f docker-compose.prod.yml up -d`.

## Endpoints

> `/admin/*` requires `Authorization: Bearer $ADMIN_TOKEN`. Public
> endpoints (`/`, `/healthz`, `/v1/*`, `/api/tags`) do not.

| Method | Path                          | Description                                  |
|--------|-------------------------------|----------------------------------------------|
| POST   | `/v1/chat/completions`        | OpenAI-compatible, proxied via rotation      |
| GET    | `/v1/models`                  | Static model list (cached)                   |
| GET    | `/api/tags`                   | Ollama-native model list (cached)            |
| POST   | `/admin/keys`                 | Create a key (returns `raw_key` once)        |
| GET    | `/admin/keys`                 | List keys (masked)                           |
| GET    | `/admin/keys/{id}`            | Get one key (masked)                         |
| PATCH  | `/admin/keys/{id}`            | Update `label` / `status`                    |
| DELETE | `/admin/keys/{id}`            | Delete a key                                 |
| POST   | `/admin/keys/{id}/test`       | Probe a single key                           |
| POST   | `/admin/keys/test-all`        | Probe all active keys                        |
| POST   | `/admin/keys/reset-states`    | Bulk-reset every key to `active`             |
| GET    | `/admin/health`               | Aggregated key-pool health (admin-gated)     |

## Configuration

See `.env.example`. Key vars: `PROBE_MODEL` (required), `ADMIN_TOKEN`
(prod, fail-closed), `OLLAMA_BASE_URL`, `DB_PATH`, `TIMEOUT`,
`MAX_ROTATION_ITERATIONS_SAFETY_MARGIN`, `MODEL_CACHE_TTL_SECONDS`,
`CORS_ALLOW_ORIGINS`, `ENABLE_DOCS`, `CADDY_DOMAIN`, `CADDY_EMAIL`.

## Security

Raw keys are returned **once** on `POST /admin/keys`. After that only the
masked prefix (`abc12345…`) is exposed. The DB stores `sha256(raw_key)`
plus an 8-char prefix; the raw value is encrypted at rest. `loguru` has
a secret-masking patcher that scrubs `key_hash` / `raw_key` /
`authorization` from logs. Admin auth uses constant-time comparison.

## Development

```bash
ruff check .
pytest tests/ -q --tb=short
python -c "import app.main; print('OK')"
```