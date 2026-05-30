# Claude Code — Project Instructions

## Language
Always respond in **English or Traditional Chinese** only. Never use Korean or Japanese.

## Git / PR Workflow
1. Develop on the designated feature branch (`claude/new-session-*` or as specified).
2. After a PR is merged, **always delete the source branch**.
   - Try `git push origin --delete <branch>` first.
   - If that errors with "remote ref does not exist", the branch was auto-deleted by GitHub on squash-merge — verify with `list_branches` and confirm it's clean. This is not an error.
3. Use squash merge for all PRs unless told otherwise.
4. Never push to `main` directly.

## Prediction Update Script
`scripts/update_prediction.py` is the canonical way to update predictions going forward.

```bash
python scripts/update_prediction.py          # update + push + open PR
python scripts/update_prediction.py --merge  # update + push + open PR + auto-merge
```

After running, delete the branch once the PR is merged (see above).
