# Numerai Signals Auto Submit

This repo runs `numerai_backup.py` via GitHub Actions once per round (Tue-Sat 13:05 UTC, with a 13:45 UTC backup).
The workflow uses `--auto` to skip if the latest live era was already submitted, and runs a cache warm-up step first.

## Setup
1. Create GitHub Actions secrets:
   - `NUMERAI_PUBLIC_ID`
   - `NUMERAI_SECRET_KEY`
   - `NUMERAI_MODEL_NAME`
2. Push to the default branch (cron only runs on the default branch).
3. Optional local test:
   ```bash
   python numerai_backup.py --run-all
   ```
   Auto mode (only submits on a new live era):
   ```bash
   python numerai_backup.py --auto
   ```
   Cache prep (download datasets + price/meta/social caches):
   ```bash
   python numerai_backup.py --prep-cache
   ```

## Notes
- The workflow runs `python numerai_backup.py --prep-cache` then `python numerai_backup.py --auto`.
- Auto submit state is stored in `cache/submission_state.json` (persisted by Actions cache).
- GitHub-hosted runners enforce a 6-hour job limit; this workflow sets `timeout-minutes` to 120.
