### Case Boundary Sentinels

Each case must emit `case_start` before the agent call and `case_end` in the `finally` block. These sentinels delimit the spans belonging to each case, giving the evaluator the ability to isolate per-case traces.

This is a codegen invariant — the generated `dispatch/<agent>.py` always emits these by construction. If you see cases without boundary sentinels in the spanlog, it indicates the harness is invoking agents outside the codegen'd dispatch modules.
