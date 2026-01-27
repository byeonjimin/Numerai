# Numerai Signals Auto Submit

This repo runs `numerai_backup.py` via GitHub Actions once per round (Tue–Sat 13:05 UTC).

## Setup
1. Create GitHub Actions secrets:
   - `NUMERAI_PUBLIC_ID`
   - `NUMERAI_SECRET_KEY`
   - `NUMERAI_MODEL_NAME`
2. Push to the default branch (cron only runs on the default branch).
3. Optional local test:
   ```bash
   python numerai_backup.py --auto-live
   ```

## Notes
- The workflow uses `--auto-live` (auto + live-only).
- GitHub-hosted runners enforce a 6-hour job limit (no workflow timeout is set here).
