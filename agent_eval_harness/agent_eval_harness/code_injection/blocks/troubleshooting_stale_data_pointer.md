### Stale Data Pointer

The plan points at a specific dataset path on disk (`{{scalar:plan.db_path}}`). If you see errors reading data, or results that mention a completely different project:

1. **First task:** {{marker:data_reachable}} — verify the data is still at that path and belongs to this target system
2. **If data changed:** Re-run the data-reachable discovery task to update the pointer
3. **If data is correct but results wrong:** The issue is not the data path — check the agent invocation itself (see RECON task in TASKS.md)
