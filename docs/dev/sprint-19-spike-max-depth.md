# Spike: Max Trace Depth Above 8

**Investigated:** Sprint 19 (May 2026)
**Question:** Can MAX_TRACE_DEPTH be raised above 8 without a pipeline stack-size change?

## Current State

- `MAX_TRACE_DEPTH = 5` (OptiXData.h, file scope — not inside `RayTracingConstants`)
- `OptixPipelineSetStackSize` called in `OptiXContext.cpp:457` with:
  - directCallableStackSizeFromState: computed by `optixUtilComputeStackSizes` (max_cc_depth = 0, max_dc_depth = 0 → effectively 0)
  - directCallableStackSizeFromTraversal: computed by `optixUtilComputeStackSizes` (same)
  - continuationStackSize: `max(optixUtilComputeStackSizes result, 49152)` — a hardcoded 48 KB floor
  - maxTraversableGraphDepth: `16` (supports recursive IAS chains up to level-14 sponges)

## Stack Budget Analysis

Stack sizing uses the OptiX utility function `optixUtilComputeStackSizes`, which computes the
continuation stack as a function of:

1. The per-shader-group stack frame sizes (derived from register and local variable usage,
   accumulated via `optixUtilAccumulateStackSizes`).
2. The `max_trace_depth` argument passed to `optixUtilComputeStackSizes` — currently set to
   `MAX_TRACE_DEPTH` (5).

The formula is: `continuationStackSize = css_CH * max_trace_depth + max(css_CH, css_MS, css_IS)`,
where css_* are the per-group continuation stack sizes (see OptiX Programming Guide, §6.3).
The utility handles the exact accounting; the important point is that the result scales linearly
with `max_trace_depth`.

The hardcoded floor of **48 KB** (`MIN_CONTINUATION_STACK_SIZE = 49152`) was added in Sprint 18
to accommodate metallic-cylinder reflection shaders. At `MAX_TRACE_DEPTH = 5` this floor is
almost certainly larger than the formula result, meaning the pipeline is currently running with
significant headroom.

To verify: at depth 5 the formula yields some value X ≤ 48 KB (otherwise the floor would not
have been added). Raising depth to 8 multiplies the formula result by 8/5 = 1.6×; if X is, say,
20–25 KB then 1.6× = 32–40 KB, still under the 48 KB floor. At depth 12 the multiplier is
12/5 = 2.4×, yielding 48–60 KB — potentially above the floor.

However, **two code paths must both change** when raising `MAX_TRACE_DEPTH`:
1. `MAX_TRACE_DEPTH` in `OptiXData.h` (the compile-time constant used by shaders).
2. `max_trace_depth` in `OptiXContext.cpp:424` — currently read from `MAX_TRACE_DEPTH`, so it
   tracks automatically.

Because `max_trace_depth` is already read from `MAX_TRACE_DEPTH` at line 424, the
`optixUtilComputeStackSizes` call already scales with any change to the constant. The 48 KB
floor absorbs the increase up to roughly depth 10–12 (exact threshold depends on actual
shader frame sizes, which require a compiled PTX inspection to measure precisely).

## Can We Raise It Without Code Changes?

**Possibly yes, up to approximately depth 10–12**, because:

- The continuation stack is not a flat value; it is computed by `optixUtilComputeStackSizes`
  using `MAX_TRACE_DEPTH`, so it grows automatically with any constant bump.
- The 48 KB floor (`MIN_CONTINUATION_STACK_SIZE`) provides extra headroom above the computed
  value; whether that headroom is consumed at depth 8, 10, or 12 depends on the actual
  per-shader frame size, which was not directly measured in this spike.
- The `max_trace_depth` variable on line 424 already reads `MAX_TRACE_DEPTH`, so the pipeline
  recalculates the stack requirement automatically.

**Risk:** If the formula result at depth 12 exceeds 48 KB, the pipeline will silently pass (the
computed value is used as-is by OptiX) but shader stack overflows on deep paths may occur at
runtime. The safe path is to raise the floor in concert with the depth.

## Recommendation

Raise `MAX_TRACE_DEPTH` to 12 in Sprint 20 with the following coordinated changes:

1. `optix-jni/src/main/native/include/OptiXData.h`: change `MAX_TRACE_DEPTH` from 5 to 12.
2. `optix-jni/src/main/native/OptiXContext.cpp`: raise `MIN_CONTINUATION_STACK_SIZE` from
   49152 (48 KB) to 98304 (96 KB) as a conservative 2× safety margin, or measure the actual
   per-shader frame size from compiled PTX (`ptxas -v`) and size the floor accordingly.
3. Update the comment on line 13 of `OptiXData.h` to reflect the new intent (e.g. "Allow
   up to 12 internal reflections/refractions").

Raising depth without the floor increase may work in practice due to existing headroom, but
making both changes together is low-risk and eliminates uncertainty.
