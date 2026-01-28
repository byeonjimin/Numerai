# Numerai Signals Auto Submit

This repo runs `numerai_signal_final.py` via GitHub Actions once per round (Tue-Sat 13:05 UTC, with a 13:45 UTC backup).

## Setup
1. Create GitHub Actions secrets:
   - `NUMERAI_PUBLIC_ID`
   - `NUMERAI_SECRET_KEY`
   - `NUMERAI_MODEL_NAME`
2. Push to the default branch (cron only runs on the default branch).
3. Optional local test:
   ```bash
   python numerai_signal_final.py --run-all
   ```

## Notes
- The workflow runs `python numerai_signal_final.py --run-all`.
- GitHub-hosted runners enforce a 6-hour job limit; this workflow sets `timeout-minutes` to 120.