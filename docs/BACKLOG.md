# Feature Backlog

Unscheduled feature ideas not yet assigned to a sprint. See [ROADMAP.md](../ROADMAP.md) for
the sprint plan and [CODE_IMPROVEMENTS.md](../CODE_IMPROVEMENTS.md) for tech-debt items.

---

## Features

### F-ND-GEOMETRY: N-Dimensional Geometry (N ≥ 5)

**Priority:** Medium
**Effort:** Large (multiple sprints)
**Dependencies:** Existing 4D projection infrastructure (Sprint 18/25), `menger-geometry` layer

**Description:**
Generalise the 4D→3D projection pipeline to support arbitrary N-dimensional geometry for N ≥ 5.
Each dimension above 3 is projected down one step at a time: 5D→4D→3D, 6D→5D→4D→3D, etc.,
using the same perspective/orthographic projection logic already used for 4D→3D.

**Parameters required per extra dimension (per N > 4):**
- N-D viewpoint: position of the observer in dimension N (analogous to `w` in 4D)
- N-D viewing distance: perspective depth for the projection from N to N-1
- N-D rotation: transformation matrix or angle set operating in dimension N

**Design notes:**
- The projection chain is composable: `projectND → project(N-1)D → … → project4D → project3D`.
  Each step is structurally identical to the existing `Project4D` kernel.
- GPU side: a single generalised kernel parameterised by dimension count, or one kernel per
  projection step chained via intermediate buffers. The former is cleaner; the latter reuses
  existing `Project4D` as-is.
- DSL: `viewpoint5d`, `viewDistance5d`, `rotation5d`, … per extra dimension, or a sequence
  `ndViewpoints: Seq[Float]`, `ndViewDistances: Seq[Float]` indexed by dimension.
- Menger sponge analogs exist in 5D (penteract sponge); same recursive IAS approach applies.
- Memory: each extra dimension adds one projection pass over all vertices; acceptable for
  moderate vertex counts.

**Suggested sprint decomposition (when scheduled):**
1. Generalise `Project4D` CUDA kernel to `ProjectND` parameterised by source dimension
2. Scala DSL + CLI parameters for 5D viewpoint/distance/rotation
3. 5D geometry: penteract (5-cube), 5D Menger sponge analog
4. Integration tests + reference images for 5D scenes
5. Extend to 6D+ (straightforward once 5D pipeline exists)
