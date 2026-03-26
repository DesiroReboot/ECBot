# ECBot Configuration Sources

## Public (tracked by git)
- `config/config.md`: variable documentation.
- `.env.example`: shared defaults and key templates.

## Private (not tracked)
- `E:\DATA\ECBot\.env` (recommended)
- You can override path with `ECBOT_DOTENV_PATH`.

## Load Order
1. Private dotenv (`ECBOT_DOTENV_PATH`, default `E:\DATA\ECBot\.env` if exists, else local `.env`)
2. Public template (`.env.example`) as fallback for missing keys
3. JSON config (`config/config.json`)

Because dotenv loading uses non-overwrite semantics, earlier sources have higher priority.

## Team Workflow
- Add new variable keys to `.env.example` and this file first.
- Keep secrets only in private dotenv.
- After pulling latest base, private dotenv continues to override while new keys are backfilled by `.env.example`.
