# Colored Transparent Shadows (Phase 1) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable colored shadow attenuation for transparent objects — a red glass sphere casts a red-tinted shadow, not a gray one — gated behind the existing `transparent_shadows_enabled` flag so all current tests pass unchanged.

**Architecture:** Expand shadow ray payload from 1 float (scalar alpha) to 3 floats (RGB attenuation). When `transparent_shadows_enabled = false`, all three channels carry the same scalar value (backward-compatible). When `true`, each channel carries `(1 - alpha) + alpha * material_color_channel`, producing colored shadow attenuation. `calculateLighting()` switches from scalar to per-channel shadow multiplication.

**Tech Stack:** CUDA/OptiX shaders (C++), Scala 3 test suite (AnyFlatSpec)

**Scope limitation (Phase 1):** Closesthit-only. Only the closest transparent object along the shadow ray contributes color. Multi-object accumulation (anyhit-based) is deferred to Phase 2.

---

## Background: Why the Previous Attempt Failed

AD-8 documents three root causes:
1. **Scope coupling** — shadow rewrite mixed with SBT offset fix (now resolved)
2. **Anyhit control flow** — `optixTerminateRay`/`optixIgnoreIntersection` changed edge-case behavior (Phase 1 avoids anyhit entirely)
3. **Scalar-to-float3 ripple** — changing `shadow_factor` from `float` to `float3` altered brightness (Phase 1 gates this behind the flag)

## Key Design Decisions

### Shadow attenuation encoding

Current (scalar): `attenuation = alpha` where 0.0 = no shadow, 1.0 = full shadow.

Colored: `attenuation_rgb = alpha * (1 - material_color_rgb)` where (0,0,0) = no shadow (fully lit), (1,1,1) = full shadow (fully blocked). A red sphere (rgb=1,0,0) with alpha=0.8 produces attenuation (0, 0.8, 0.8) — blocks green and blue, passes red. The shadow factor (what multiplies the light contribution) is `1 - attenuation_rgb = (1, 0.2, 0.2)`.

When `transparent_shadows_enabled = false`: all three channels are set to the scalar `alpha`, producing `shadow_factor = (1-alpha, 1-alpha, 1-alpha)` — mathematically identical to the current scalar path.

### Why closesthit is sufficient for Phase 1

The closesthit shadow shader sees the closest intersected object. For the common case (one transparent object between light and surface), this is the only object. The limitation — ignoring objects behind the closest — is acceptable for Phase 1 and matches what many production renderers do for a first pass.

---

## Task 1: Expand shadow payload to 3 channels in shadow shaders

**Files:**
- Modify: `optix-jni/src/main/native/shaders/shadows.cu` (sphere shadow CH + miss)
- Modify: `optix-jni/src/main/native/shaders/hit_triangle.cu:302-312` (triangle shadow CH)
- Modify: `optix-jni/src/main/native/shaders/hit_cylinder.cu:314-324` (cylinder shadow CH)

### Step 1: Update `__miss__shadow()` to set 3 payload slots

In `shadows.cu`, change from:
```cpp
extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(__float_as_uint(0.0f));
}
```
To:
```cpp
extern "C" __global__ void __miss__shadow() {
    // No occlusion — zero attenuation on all channels
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}
```

### Step 2: Update `__closesthit__shadow()` (sphere) for colored attenuation

In `shadows.cu`, change from:
```cpp
extern "C" __global__ void __closesthit__shadow() {
    float4 material_color;
    float material_ior;
    getInstanceMaterial(material_color, material_ior);
    const float alpha = material_color.w;
    optixSetPayload_0(__float_as_uint(alpha));
}
```
To:
```cpp
extern "C" __global__ void __closesthit__shadow() {
    float4 material_color;
    float material_ior;
    getInstanceMaterial(material_color, material_ior);
    const float alpha = material_color.w;

    if (params.transparent_shadows_enabled) {
        // Colored attenuation: how much of each channel is BLOCKED
        // A red sphere (rgb=1,0,0) with alpha=0.8 blocks 0% red, 80% green, 80% blue
        optixSetPayload_0(__float_as_uint(alpha * (1.0f - material_color.x)));
        optixSetPayload_1(__float_as_uint(alpha * (1.0f - material_color.y)));
        optixSetPayload_2(__float_as_uint(alpha * (1.0f - material_color.z)));
    } else {
        // Scalar mode: uniform attenuation across all channels (backward-compatible)
        optixSetPayload_0(__float_as_uint(alpha));
        optixSetPayload_1(__float_as_uint(alpha));
        optixSetPayload_2(__float_as_uint(alpha));
    }
}
```

### Step 3: Update `__closesthit__triangle_shadow()` (same pattern)

In `hit_triangle.cu:302-312`, apply the same pattern as Step 2.

### Step 4: Update `__closesthit__cylinder_shadow()` (same pattern)

In `hit_cylinder.cu:314-324`, apply the same pattern as Step 2. Also fix the M-shadow-material-inconsistency: change `getInstanceMaterialPBR(...)` to `getInstanceMaterial(material_color, material_ior)` since only alpha is needed.

### Step 5: Compile and verify no build errors

Run: `sbt "project optixJni" nativeCompile`
Expected: SUCCESS (no new symbols, just expanded payload usage)

### Step 6: Commit

```
feat(shaders): expand shadow payload to RGB in closesthit shaders

Shadow closesthit programs now set payloads 0-2 (R, G, B attenuation).
When transparent_shadows_enabled=false, all three channels carry the same
scalar alpha value (backward-compatible). When true, each channel encodes
per-wavelength attenuation based on material color.
```

---

## Task 2: Update traceShadowRay() to return float3

**Files:**
- Modify: `optix-jni/src/main/native/shaders/helpers.cu:31-74` (traceShadowRay)
- Modify: `optix-jni/src/main/native/shaders/helpers.cu:150-200` (calculateLighting)

### Step 1: Change traceShadowRay return type and unpack 3 payloads

In `helpers.cu:31-74`, change from:
```cpp
__device__ float traceShadowRay(
    const float3& hit_point,
    const float3& normal,
    const float3& light_dir
) {
    const float3 shadow_origin = hit_point + normal * SHADOW_RAY_OFFSET;
    unsigned int shadow_payload = 0;

    optixTrace(
        params.handle,
        shadow_origin,
        light_dir,
        SHADOW_RAY_OFFSET,
        SHADOW_RAY_MAX_DISTANCE,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        params.sbt_base_offset + SBTConstants::RAY_TYPE_SHADOW,
        SBTConstants::STRIDE_RAY_TYPES,
        SBTConstants::MISS_SHADOW,
        shadow_payload
    );

    const float shadow_attenuation = __uint_as_float(shadow_payload);
    const float shadow_factor = SBTConstants::SHADOW_FACTOR_FULLY_LIT - shadow_attenuation;

    if (params.stats) {
        atomicAdd(&params.stats->shadow_rays, 1ULL);
        atomicAdd(&params.stats->total_rays, 1ULL);
    }

    return shadow_factor;
}
```
To:
```cpp
__device__ float3 traceShadowRay(
    const float3& hit_point,
    const float3& normal,
    const float3& light_dir
) {
    const float3 shadow_origin = hit_point + normal * SHADOW_RAY_OFFSET;
    unsigned int shadow_p0 = 0, shadow_p1 = 0, shadow_p2 = 0;

    optixTrace(
        params.handle,
        shadow_origin,
        light_dir,
        SHADOW_RAY_OFFSET,
        SHADOW_RAY_MAX_DISTANCE,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        params.sbt_base_offset + SBTConstants::RAY_TYPE_SHADOW,
        SBTConstants::STRIDE_RAY_TYPES,
        SBTConstants::MISS_SHADOW,
        shadow_p0, shadow_p1, shadow_p2
    );

    // Unpack per-channel shadow attenuation and convert to shadow factor
    // attenuation 0.0 = fully lit, 1.0 = fully blocked
    // shadow_factor 1.0 = fully lit, 0.0 = fully blocked
    const float3 shadow_factor = make_float3(
        SBTConstants::SHADOW_FACTOR_FULLY_LIT - __uint_as_float(shadow_p0),
        SBTConstants::SHADOW_FACTOR_FULLY_LIT - __uint_as_float(shadow_p1),
        SBTConstants::SHADOW_FACTOR_FULLY_LIT - __uint_as_float(shadow_p2)
    );

    if (params.stats) {
        atomicAdd(&params.stats->shadow_rays, 1ULL);
        atomicAdd(&params.stats->total_rays, 1ULL);
    }

    return shadow_factor;
}
```

### Step 2: Update calculateLighting() to use float3 shadow_factor

In `helpers.cu:150-200`, change the shadow_factor usage from:
```cpp
        const float shadow_factor = (params.shadows_enabled && !skip_shadows)
            ? traceShadowRay(hit_point, normal, light_dir)
            : SBTConstants::SHADOW_FACTOR_FULLY_LIT;

        // ...
        total_lighting = total_lighting + light_color * light.intensity * attenuation * ndotl * shadow_factor;
```
To:
```cpp
        const float3 shadow_factor = (params.shadows_enabled && !skip_shadows)
            ? traceShadowRay(hit_point, normal, light_dir)
            : make_float3(SBTConstants::SHADOW_FACTOR_FULLY_LIT,
                          SBTConstants::SHADOW_FACTOR_FULLY_LIT,
                          SBTConstants::SHADOW_FACTOR_FULLY_LIT);

        // Per-channel shadow attenuation: colored shadows tint the light contribution
        const float3 shadowed_contribution = make_float3(
            light_color.x * light.intensity * attenuation * ndotl * shadow_factor.x,
            light_color.y * light.intensity * attenuation * ndotl * shadow_factor.y,
            light_color.z * light.intensity * attenuation * ndotl * shadow_factor.z
        );
        total_lighting = total_lighting + shadowed_contribution;
```

### Step 3: Compile and verify no build errors

Run: `sbt "project optixJni" nativeCompile`
Expected: SUCCESS

### Step 4: Run full test suite — all existing tests must pass

Run: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt test`
Expected: All 2044 tests pass. The `transparent_shadows_enabled` flag defaults to `false`, so all shadow shaders write identical values to all 3 payload channels, and `calculateLighting()` produces the same result as before.

**CRITICAL:** If any test fails here, STOP and investigate per TEST FAILURE PROTOCOL. The flag is off, so behavior must be identical. A failure means a bug in the implementation, not a test issue.

### Step 5: Commit

```
feat(shaders): return RGB shadow factor from traceShadowRay

traceShadowRay() now returns float3 (per-channel shadow factor) instead
of float. calculateLighting() applies per-channel shadow multiplication.
When transparent_shadows_enabled=false (default), all channels carry the
same scalar value, producing identical results to the previous scalar path.
```

---

## Task 3: Write tests for colored shadow behavior

**Files:**
- Create: `optix-jni/src/test/scala/menger/optix/ColoredShadowSuite.scala`

### Step 1: Write ColoredShadowSuite with tests that exercise the colored shadow path

Tests to implement:

1. **"Colored shadows should be disabled by default"** — render with default config, verify behavior matches non-transparent shadows render (images equal).

2. **"Enabling colored shadows should not crash"** — `noException should be thrownBy` render with `transparentShadows = true`.

3. **"Red transparent sphere should cast shadow with reduced red attenuation"** — render red sphere (rgba 1,0,0,0.8) with colored shadows on, sample shadow region. The shadow should have a red tint: the R channel in the shadow region should be brighter (less attenuated) than G and B channels.

4. **"Blue transparent sphere should cast blue-tinted shadow"** — same as above with blue sphere, B channel brighter in shadow.

5. **"Green transparent sphere should cast green-tinted shadow"** — same with green.

6. **"White transparent sphere should cast achromatic shadow (same as scalar)"** — white sphere (1,1,1,alpha) should produce shadow where R≈G≈B (no color tint), consistent with scalar shadow behavior.

7. **"Fully opaque sphere should cast full shadow regardless of color"** — with alpha=1.0, shadow attenuation should be (1,1,1) regardless of color. Shadow region should be dark with no color tint.

8. **"Fully transparent sphere should cast no shadow regardless of color"** — with alpha=0.0, shadow factor should be (1,1,1). Shadow region brightness should equal no-sphere brightness.

9. **"Colored shadow disabled vs enabled should differ for colored transparent sphere"** — render same red transparent sphere with flag on vs off, images should differ (the shadow region will have different color balance).

10. **"Colored shadow disabled vs enabled should be identical for white sphere"** — white sphere produces identical results with flag on or off (since all channels attenuate equally).

### Step 2: Run new tests to verify they fail (TDD: red phase)

Run: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt "testOnly menger.optix.ColoredShadowSuite"`
Expected: Tests 1, 2, 6, 7, 8, 10 may pass already (they test boundary conditions). Tests 3, 4, 5, 9 should fail (colored tint not yet visible because `transparent_shadows_enabled` was a no-op... but wait, Task 2 already implemented the shader logic). Actually, since Task 2 wired the flag through to the shaders, tests that enable the flag should already produce colored shadows. So all tests should pass.

Re-think: This is not classic TDD because the shader implementation (Tasks 1-2) must exist before tests can exercise it (you can't partially implement a shader). The tests in Task 3 are validation tests, not driving tests. Write them after the shader changes.

### Step 3: Run the tests

Run: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt "testOnly menger.optix.ColoredShadowSuite"`
Expected: All pass

### Step 4: Run full test suite to verify no regressions

Run: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt test`
Expected: All 2044 + new tests pass

### Step 5: Commit

```
test(shadows): add ColoredShadowSuite for RGB shadow attenuation

Tests verify: colored tint for R/G/B spheres, achromatic shadow for
white spheres, boundary cases (alpha=0, alpha=1), flag on/off parity
for white spheres, and divergence for colored spheres.
```

---

## Task 4: Clean up dead code from the reverted attempt

**Files:**
- Modify: `optix-jni/src/main/native/shaders/hit_cylinder.cu:294-336` (remove dead anyhit programs)
- Modify: `optix-jni/src/main/native/OptiXContext.cpp:322-354` (remove unused anyhit overload)
- Modify: `optix-jni/src/main/native/include/OptiXContext.h:79-84` (remove declaration)
- Modify: `CODE_IMPROVEMENTS.md` (mark H-dead-anyhit, H-dead-overload, H-transparent-shadows-dead as resolved)

### Step 1: Remove `__anyhit__cylinder()` and `__anyhit__cylinder_shadow()` from hit_cylinder.cu

These are never registered in any hitgroup (confirmed by PipelineManager.cpp inspection).

### Step 2: Remove the 3-parameter `createTriangleHitgroupProgramGroup` overload

The CH+AH overload in OptiXContext.cpp:322-354 and its declaration in OptiXContext.h are never called.

### Step 3: Remove the 6-parameter `createHitgroupProgramGroup` overload

The CH+AH+IS overload in OptiXContext.cpp:255-288 and its declaration in OptiXContext.h are also never called. (Verify this claim before removing.)

### Step 4: Update CODE_IMPROVEMENTS.md

Mark H-dead-anyhit, H-dead-overload as resolved. Update H-transparent-shadows-dead: the field is now read by shaders, so it's no longer dead code — mark resolved.

### Step 5: Compile and run full test suite

Run: `sbt "project optixJni" nativeCompile`
Run: `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt test`
Expected: All tests pass (removed code was unreachable)

### Step 6: Commit

```
refactor(shaders): remove dead anyhit programs and unused overloads

Remove __anyhit__cylinder, __anyhit__cylinder_shadow (never registered
in any hitgroup), and unused createTriangleHitgroupProgramGroup and
createHitgroupProgramGroup anyhit overloads (never called).

Resolves: H-dead-anyhit, H-dead-overload, H-transparent-shadows-dead
```

---

## Task 5: Update documentation

**Files:**
- Modify: `CHANGELOG.md` (add colored shadows feature)
- Modify: `docs/arc42/11-risks-and-technical-debt.md` (update TD-6 status)
- Modify: `docs/arc42/08-crosscutting-concepts.md` (document colored shadow algorithm if applicable)

### Step 1: Update CHANGELOG.md

Add under `[Unreleased]` → `Added`:
- Colored transparent shadows: transparent objects cast color-tinted shadows when `--transparent-shadows` is enabled (Phase 1: single-object, closesthit-only)

### Step 2: Update TD-6

Change status from "Medium" to "Low" — Phase 1 is implemented, Phase 2 (multi-object anyhit) remains as reduced-scope debt.

### Step 3: Commit

```
docs: update changelog and technical debt for colored shadows Phase 1
```

---

## Task 6: Update shell test scripts

Per AGENTS.md, every new rendering feature must be covered in both shell scripts.

**Files:**
- Modify: `scripts/integration-tests.sh` — add `test_colored_shadows()` function
- Modify: `scripts/manual-test.sh` — add static and interactive entries

### Step 1: Add integration test

Add a `test_colored_shadows()` function that renders a red transparent sphere with `--transparent-shadows` and verifies the output image differs from a render without the flag.

### Step 2: Add manual test entries

Add static renders showing colored shadows for red, green, blue transparent spheres. Add an interactive entry for exploring colored shadow effects.

### Step 3: Run integration tests

Run: `scripts/integration-tests.sh`
Expected: All tests pass including new one

### Step 4: Commit

```
test(integration): add colored shadows to shell test scripts
```

---

## Verification Checklist (before declaring complete)

- [ ] `__GL_THREADED_OPTIMIZATIONS=0 xvfb-run sbt test` — all tests pass (old + new)
- [ ] `sbt "scalafix --check"` — passes
- [ ] Shell integration tests pass
- [ ] `transparent_shadows_enabled = false` produces byte-identical images to pre-change code
- [ ] Red/green/blue transparent spheres produce visibly tinted shadows when flag is on
- [ ] White transparent sphere produces identical shadow with flag on vs off
- [ ] No anyhit programs are involved (Phase 1 is closesthit-only)
- [ ] CHANGELOG, arc42 docs, CODE_IMPROVEMENTS.md are all updated
