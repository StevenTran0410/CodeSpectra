### Anchor Validation Failed

The expected anchor lines for the main entrypoint are missing or have drifted. This prevents the plan's marker-block edits from applying cleanly.

**Check that:**
- You are on the correct branch (`{{scalar:plan.branch_name}}`)
- Your main entrypoint file has not been substantially refactored since the plan was generated
- If in doubt, re-run the discovery step to regenerate anchors for the current codebase state
