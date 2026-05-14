# Spike Findings: Fractional Levels with IAS Sponges

**Sprint:** 19.10  
**Date:** 2026-05-14  
**Status:** Complete — recommendation issued

---

## Question

Can fractional sponge levels (alpha-blended transition between integer levels) be applied
to `sponge-recursive-ias`? The tessellated sponge supports fractional levels via per-vertex
alpha. Does the IAS approach have an equivalent?

---

## Current IAS Architecture

| Component | File | Notes |
|-----------|------|-------|
| JNI interface | `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala:586–732` | `addRecursiveIASSpongeInstance` |
| C++ impl | `optix-jni/src/main/native/OptiXWrapper.cpp:1667–1845` | `buildSubIAS`, `getMengerGenerators` |
| Leaf geometry | `OptiXWrapper.cpp:1796–1806` | most-recently-uploaded unit cube GAS |
| Scene integration | `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala:104–113` | |
| Tests | `optix-jni/src/test/scala/menger/optix/RecursiveIASSpongeSuite.scala` | single-instance smoke tests |

**Nesting:** Each recursive IAS at level N wraps 20 sub-IAS (Menger generators,
hardcoded 4×3 transforms). Max 14 levels (`MAX_RECURSIVE_IAS_LEVEL = 14`,
`OptiXWrapper.cpp:1690`) — OptiX `MAX_TRAVERSABLE_GRAPH_DEPTH = 16` is the hard ceiling.

**Material model:** Each top-level instance has its own `ObjectInstance { color[4], ior, ... }`
(`ObjectInstance` struct, `OptiXWrapper.cpp:72–90`). All 20 nested sub-IAS children inherit
the parent's `instanceId` — **no per-sub-cube material differentiation within a single tree.**

---

## Approach A — Per-leaf Alpha via SBT Instance Records

Modify the hit shader to read alpha from per-SBT-instance records rather than the shared
`ObjectInstance` array. Each of the 20^N leaf cubes would carry its own alpha.

| Property | Value |
|----------|-------|
| VRAM | O(20^N) — level 5 = 3.2M SBT records |
| Complexity | High — SBT architecture changes |
| Verdict | **Not viable.** Eliminates the core VRAM advantage of IAS. |

---

## Approach B — Two IAS Trees at Adjacent Integer Levels

Render two trees simultaneously for fractional level N+f (0 < f < 1):

- **Tree 1:** Level N at opacity 1.0 — the fine structure, fully opaque
- **Tree 2:** Level N−1 at transparency f → opacity (1−f) — the coarser skin, fading out

**Visual semantics:**

| f | Tree 2 (N−1) opacity | Result |
|---|----------------------|--------|
| 0 | 1.0 | Level N−1 big cubes fully occlude level N interior → looks like level N−1 |
| →1 | →0 | Level N−1 becomes transparent, revealing level N beneath |
| 1 | 0 | Invisible → only level N visible → looks exactly like integer level N |

The transition is correct: level N−1 acts as a fading outer skin that gradually unmasks
the finer level N structure.

**Overlapping transparent IAS — investigated in this spike:**

Two overlapping `sponge-recursive-ias` instances can coexist and render correctly:

- **Instance budget:** 64 instances per scene (`MAX_INSTANCES`, `OptiXData.h:54`). Two
  level-N sponges consume ~40 slots — within budget for levels ≤ 13.
- **Ray traversal:** OptiX traces rays through both IAS volumes in depth order. Per-instance
  `InstanceMaterial` is indexed via `optixGetInstanceId()` (`hit_triangle.cu:197`) —
  each tree gets its own independent alpha.
- **Transparent shadows:** `accumulateShadowAttenuation()` (`helpers.cu`) accumulates
  attenuation through overlapping transparent geometry — physically correct.
- **No scene restriction:** `TriangleMeshSceneBuilder` validates total instance count,
  not sponge uniqueness. Multiple recursive-IAS sponges are supported today.

**No architectural blocker. Two overlapping transparent IAS sponges render correctly with
the existing pipeline.**

**Cost/complexity:**

| Property | Value |
|----------|-------|
| VRAM | 2× one level's IAS memory — still O(N), not O(20^N) |
| API change | Two `addRecursiveIASSpongeInstance` calls with different level and opacity |
| Shader change | None — per-instance alpha already in `ObjectInstance.color[3]` |
| Compositor change | None — ray tracing handles overlapping volumes natively |
| Correctness at integers | Tree 2 opacity = 0 → invisible → exact match with single tree |

---

## Recommendation

**Approach B — implement in Sprint 21.**

Approach A is ruled out (VRAM explosion, defeats IAS purpose). Approach B is
architecturally clean: no shader changes, no compositor changes, two API calls.
The overlapping-IAS question is resolved — the existing pipeline handles it correctly.

**Sprint 21 work items:**
1. Extend `addRecursiveIASSpongeInstance` DSL to accept a `fractionalLevel: Float`
   parameter and internally issue two renderer calls (level `ceil(f)` + level `floor(f)`
   at computed opacity).
2. Add a smoke-test rendering two overlapping IAS sponges at fractional level to
   `RecursiveIASSpongeSuite`.
