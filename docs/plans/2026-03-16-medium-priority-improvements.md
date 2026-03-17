# Medium Priority Code Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all open medium-priority issues from CODE_IMPROVEMENTS.md and close already-resolved items.

**Architecture:** This is a mixed refactoring plan: documentation fixes, dead-code removal in C++ shader headers, and housekeeping. No Scala business logic changes. Shader constant changes require CUDA compilation to validate.

**Tech Stack:** Scala 3, CUDA/C++ (OptiX shaders), SBT build system

---

## Pre-Work: Orientation

Before starting, understand the scope:
- `docs/USER_GUIDE.md` — documentation file
- `optix-jni/src/main/native/shaders/` — GPU shader files (compiled via CMakeLists.txt)
- `optix-jni/src/main/native/include/OptiXData.h` — shared header with constant namespaces
- Test command: `sbt test` (runs ~1,070 tests: 27 C++, 1,043 Scala)
- Chain commands with `;` not `&&`

---

## Task 1: Close Already-Resolved Issues in CODE_IMPROVEMENTS.md

**Files:**
- Modify: `CODE_IMPROVEMENTS.md`

Investigation has confirmed the following Medium Priority issues are **already fixed in code**. They need to be removed from CODE_IMPROVEMENTS.md and moved to the resolved section.

### Pre-check (read before editing)

Verify each is indeed resolved:

1. **M-shadow-material-inconsistency** — `hit_cylinder.cu:298` `__closesthit__cylinder_shadow()` already calls `getInstanceMaterial` (not `getInstanceMaterialPBR`). Both shadow shaders match. ✓ Resolved.

2. **M-eyew-dup** — `InputHandler.scala:64-65` has `computeEyeW` in `CameraHandler` trait. Both handlers call it. ✓ Resolved.

3. **M-key-dup** — `KeyRotation.scala` has `factor: Map[Key, Int]` and `angle()` shared between both key handlers. ✓ Resolved.

4. **M-btn** — `LibGDXConverters.scala:37` has `toGdxButton`. `GdxCameraHandler` calls `LibGDXConverters.toGdxButton`. ✓ Resolved.

5. **M-userguide-t-animation-version** — `USER_GUIDE.md:911` reads "**Introduced in v0.5.2**". ✓ Resolved.

6. **M-userguide-deprecated-flags** — `USER_GUIDE.md:1787` already uses `--objects 'type=sphere:ior=1.5:size=1.5'`. Section 8.2 is correct. ✓ Resolved.

   > **Note:** Separately, section 6.2 "Custom Materials" (line 628–631) and the caustics example (line 1045) still reference `--ior 1.5` as a standalone top-level flag that was removed in v0.4.3. This is a new untracked issue. Add a new low-priority note for it.

### Step 1: Edit CODE_IMPROVEMENTS.md

Move the six resolved items from Medium Priority to the "Accepted / Deferred" table (or a "Resolved" table if you prefer). Also add a new low-priority item for the remaining `--ior` in sections 6.2 and the caustics example.

The resolved items to remove from Medium Priority:
- `M-shadow-material-inconsistency`
- `M-eyew-dup`
- `M-key-dup`
- `M-btn`
- `M-userguide-t-animation-version`
- `M-userguide-deprecated-flags`

For each, add to the existing `Accepted / Deferred` table at the bottom with `| Item | Decision |` format, noting "Resolved in Sprint N" or similar.

New low-priority item to add under `## Low Priority`:

```markdown
### L-userguide-ior-flag-stale — Sections 6.2 and caustics examples use removed --ior flag
**Location:** `docs/USER_GUIDE.md` section 6.2 (lines ~628–631), caustics tutorial (~line 1045)
**Est. Effort:** 0.1h
Section 6.2 "Custom Materials" shows `--ior 1.5` as a standalone CLI flag. The caustics
tutorial example also passes `--ior 1.5` top-level. Both `--ior` and `--radius` were
removed in v0.4.3. These examples should use `ior=1.5` inside `--objects` syntax.
```

### Step 2: Commit

```bash
git add CODE_IMPROVEMENTS.md
git commit -m "docs: close resolved medium-priority issues M-shadow-material-inconsistency, M-eyew-dup, M-key-dup, M-btn, M-userguide-t-animation-version, M-userguide-deprecated-flags"
```

---

## Task 2: M-userguide-version-header — Update USER_GUIDE.md Version

**Files:**
- Modify: `docs/USER_GUIDE.md` line 3
- Modify: `CODE_IMPROVEMENTS.md` (close this item)

### Context

- `USER_GUIDE.md` line 3: `**Version**: 0.5.2`
- `USER_GUIDE.md` line 4: `**Last Updated**: March 2026` ← already correct
- Current version in `MengerCLIOptions.scala:31`: `"menger v0.5.3"`
- Only the version string needs updating; the date is already correct.

### Step 1: Fix version in USER_GUIDE.md

Change line 3 from:
```
**Version**: 0.5.2
```
to:
```
**Version**: 0.5.3
```

### Step 2: Close item in CODE_IMPROVEMENTS.md

Remove the `M-userguide-version-header` entry from the Medium Priority section and add to the resolved table.

### Step 3: Verify no other version strings need updating

```bash
grep -r "0\.5\.2" docs/USER_GUIDE.md
```
Expected: no remaining `0.5.2` version references (IOR examples etc. are fine, those aren't version strings).

### Step 4: Commit

```bash
git add docs/USER_GUIDE.md CODE_IMPROVEMENTS.md
git commit -m "docs: update USER_GUIDE.md version to 0.5.3 (was stale at 0.5.2)"
```

---

## Task 3: M-legacy-shaders — Delete Orphaned Shader Files

**Files:**
- Delete: `optix-jni/src/main/native/shaders/sphere_combined.cu`
- Delete: `optix-jni/src/main/native/shaders/sphere_raygen.cu`
- Delete: `optix-jni/src/main/native/shaders/sphere_miss.cu`
- Delete: `optix-jni/src/main/native/shaders/sphere_closesthit.cu`
- Modify: `CODE_IMPROVEMENTS.md` (close this item)

### Context

- `CMakeLists.txt` compiles only `shaders/optix_shaders.cu`. None of the four sphere_*.cu files are referenced.
- The files contain an older standalone implementation with their own `Params`, hardcoded magic numbers, and `MissData` fields that no longer exist.
- They are safe to delete.

### Step 1: Verify not referenced

```bash
grep -r "sphere_combined\|sphere_raygen\|sphere_miss\|sphere_closesthit" optix-jni/src/main/native/CMakeLists.txt
```
Expected: no output (confirmed not compiled).

### Step 2: Delete the files

```bash
git rm optix-jni/src/main/native/shaders/sphere_combined.cu
git rm optix-jni/src/main/native/shaders/sphere_raygen.cu
git rm optix-jni/src/main/native/shaders/sphere_miss.cu
git rm optix-jni/src/main/native/shaders/sphere_closesthit.cu
```

### Step 3: Run tests to confirm nothing broke

```bash
sbt test
```
Expected: all tests pass (these files were not compiled).

### Step 4: Close item in CODE_IMPROVEMENTS.md

Remove `M-legacy-shaders` from Medium Priority; add to resolved table.

### Step 5: Commit

```bash
git add CODE_IMPROVEMENTS.md
git commit -m "chore: remove orphaned legacy sphere shader files (M-legacy-shaders)

sphere_combined.cu, sphere_raygen.cu, sphere_miss.cu, sphere_closesthit.cu were
never compiled (absent from CMakeLists.txt) and referenced removed MissData fields."
```

---

## Task 4: M-color-byte-max-dup — Consolidate COLOR_BYTE_MAX

**Files:**
- Modify: `optix-jni/src/main/native/include/OptiXData.h`
- Modify: `optix-jni/src/main/native/shaders/helpers.cu`
- Modify: `CODE_IMPROVEMENTS.md`

### Context

`COLOR_BYTE_MAX = 255.0f` is defined twice:
- `RayTracingConstants::COLOR_BYTE_MAX` (line 25) — correct home for this constant
- `RenderingConstants::COLOR_BYTE_MAX` (line 131) — duplicate; used in helpers.cu:1053-1055

The fix is: remove the duplicate from `RenderingConstants` and update usages in `helpers.cu` to use `RayTracingConstants::COLOR_BYTE_MAX`.

### Step 1: Find all usages of RenderingConstants::COLOR_BYTE_MAX

```bash
grep -rn "RenderingConstants::COLOR_BYTE_MAX" optix-jni/src/main/native/
```
Expected: 3 matches in helpers.cu (lines ~1053–1055).

### Step 2: Update helpers.cu

In `helpers.cu` lines ~1053–1055, replace each `RenderingConstants::COLOR_BYTE_MAX` with `RayTracingConstants::COLOR_BYTE_MAX`.

The lines look like:
```cpp
unsigned int r = static_cast<unsigned int>(fminf(... , RenderingConstants::COLOR_BYTE_MAX));
unsigned int g = static_cast<unsigned int>(fminf(... , RenderingConstants::COLOR_BYTE_MAX));
unsigned int b = static_cast<unsigned int>(fminf(... , RenderingConstants::COLOR_BYTE_MAX));
```
Change all three to use `RayTracingConstants::COLOR_BYTE_MAX`.

### Step 3: Remove the duplicate from OptiXData.h

In `OptiXData.h`, inside `namespace RenderingConstants`, remove this line:
```cpp
constexpr float COLOR_BYTE_MAX = 255.0f;              // Maximum color component for output
```

### Step 4: Verify no remaining uses of RenderingConstants::COLOR_BYTE_MAX

```bash
grep -rn "RenderingConstants::COLOR_BYTE_MAX" optix-jni/src/main/native/
```
Expected: no output.

### Step 5: Compile to verify

```bash
sbt compile
```
Expected: clean compile.

### Step 6: Run tests

```bash
sbt test
```
Expected: all tests pass.

### Step 7: Close item in CODE_IMPROVEMENTS.md

### Step 8: Commit

```bash
git add optix-jni/src/main/native/include/OptiXData.h optix-jni/src/main/native/shaders/helpers.cu CODE_IMPROVEMENTS.md
git commit -m "refactor: consolidate COLOR_BYTE_MAX to RayTracingConstants (M-color-byte-max-dup)

Removed duplicate RenderingConstants::COLOR_BYTE_MAX (= 255.0f).
Updated helpers.cu to use RayTracingConstants::COLOR_BYTE_MAX."
```

---

## Task 5: M-naming-constants — Clean Up Overly Literal Constants in RenderingConstants

**Files:**
- Modify: `optix-jni/src/main/native/include/OptiXData.h`
- Modify: `optix-jni/src/main/native/shaders/helpers.cu`
- Modify: `optix-jni/src/main/native/shaders/raygen_primary.cu`
- Modify: `CODE_IMPROVEMENTS.md`

### Context

`RenderingConstants` in `OptiXData.h` contains two categories of problematic constants:

**Category A — Dead (never used in any shader):**
- `COLOR_WHITE = 1.0f`
- `FRESNEL_BASE = 1.0f`
- `METALLIC_THRESHOLD = 0.0f`
- `DOT_PRODUCT_CLAMP_MIN = 0.0f`
- `DOT_PRODUCT_CLAMP_MIN_SINGLE = 0.0f`

**Category B — Used but purely literal (add indirection without encoding domain meaning):**
- `COLOR_BLACK = 0.0f` → `make_float3(0.f, 0.f, 0.f)` directly is clearer
- `UNIT_CONVERSION_FACTOR = 1.0f` → just `1.0f`
- `DISTANCE_FALLOFF_NONE = 1.0f` → just `1.0f` (keep the comment inline)
- `DISTANCE_FALLOFF_BASE = 1.0f` → just `1.0f` (keep the comment inline)
- `DOT_PRODUCT_ZERO_THRESHOLD = 0.0f` → just `0.0f`
- `VACUUM_IOR = 1.0f` → use `MaterialConstants::IOR_VACUUM` (meaningful alias already exists)
- `FRESNEL_ONE_MINUS_COS = 1.0f` → just `1.0f` (the `1` in `1.0f - cos_theta` needs no constant)
- `FRESNEL_ONE_MINUS_R0 = 1.0f` → just `1.0f`

**Keep (encode real domain meaning):**
- `PIXEL_CENTER_OFFSET = 0.5f`
- `NDC_SCALE = 2.0f`
- `NDC_OFFSET = 1.0f`
- `REFLECTION_SCALE = 2.0f`
- `DIFFUSE_BLEND_FACTOR` (computed from AMBIENT_LIGHT_FACTOR)

### Step 1: Find all usages of category A+B constants

```bash
grep -rn "COLOR_BLACK\|UNIT_CONVERSION_FACTOR\|DISTANCE_FALLOFF_NONE\|DISTANCE_FALLOFF_BASE\|DOT_PRODUCT_ZERO_THRESHOLD\|VACUUM_IOR\|FRESNEL_ONE_MINUS_COS\|FRESNEL_ONE_MINUS_R0" optix-jni/src/main/native/shaders/
```

Expected matches:
- `helpers.cu:116` — `DISTANCE_FALLOFF_NONE`
- `helpers.cu:138` — `DISTANCE_FALLOFF_BASE` (×2)
- `helpers.cu:155` — `DOT_PRODUCT_ZERO_THRESHOLD`
- `helpers.cu:177` — `COLOR_BLACK` (×3 in make_float3)
- `helpers.cu:704-705` — `VACUUM_IOR` (×2)
- `helpers.cu:709` — `FRESNEL_ONE_MINUS_COS`
- `helpers.cu:710` — `FRESNEL_ONE_MINUS_R0`
- `raygen_primary.cu:39-40` — `UNIT_CONVERSION_FACTOR` (×2)

### Step 2: Update helpers.cu — replace Category B usages

**`helpers.cu:116`** — `DISTANCE_FALLOFF_NONE`:
```cpp
// Before:
attenuation = RenderingConstants::DISTANCE_FALLOFF_NONE;  // No distance falloff
// After:
attenuation = 1.0f;  // No distance falloff for directional lights
```

**`helpers.cu:138`** — `DISTANCE_FALLOFF_BASE` (×2):
```cpp
// Before:
attenuation = RenderingConstants::DISTANCE_FALLOFF_BASE / (RenderingConstants::DISTANCE_FALLOFF_BASE + distance * distance);
// After:
attenuation = 1.0f / (1.0f + distance * distance);  // Inverse-square law: I = I₀/(1+d²)
```

**`helpers.cu:155`** — `DOT_PRODUCT_ZERO_THRESHOLD`:
```cpp
// Before:
return fmaxf(RenderingConstants::DOT_PRODUCT_ZERO_THRESHOLD, dot(normal, light_dir));
// After:
return fmaxf(0.0f, dot(normal, light_dir));
```

**`helpers.cu:177`** — `COLOR_BLACK` (×3 in make_float3):
```cpp
// Before:
float3 total_lighting = make_float3(RenderingConstants::COLOR_BLACK, RenderingConstants::COLOR_BLACK, RenderingConstants::COLOR_BLACK);
// After:
float3 total_lighting = make_float3(0.f, 0.f, 0.f);
```

**`helpers.cu:704-705`** — `VACUUM_IOR` (×2):
```cpp
// Before:
const float n1 = entering ? RenderingConstants::VACUUM_IOR : material_ior;
const float n2 = entering ? material_ior : RenderingConstants::VACUUM_IOR;
// After:
const float n1 = entering ? MaterialConstants::IOR_VACUUM : material_ior;
const float n2 = entering ? material_ior : MaterialConstants::IOR_VACUUM;
```

**`helpers.cu:709-710`** — `FRESNEL_ONE_MINUS_COS` and `FRESNEL_ONE_MINUS_R0`:
```cpp
// Before:
const float one_minus_cos = RenderingConstants::FRESNEL_ONE_MINUS_COS - cos_theta;
return R0 + RenderingConstants::FRESNEL_ONE_MINUS_R0 * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
// After:
const float one_minus_cos = 1.0f - cos_theta;
return R0 + one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos * one_minus_cos;
```

### Step 3: Update raygen_primary.cu — replace UNIT_CONVERSION_FACTOR

**`raygen_primary.cu:39-40`**:
```cpp
// Before:
const float pixel_half_width = RenderingConstants::UNIT_CONVERSION_FACTOR / static_cast<float>(dim.x);
const float pixel_half_height = RenderingConstants::UNIT_CONVERSION_FACTOR / static_cast<float>(dim.y);
// After:
const float pixel_half_width = 1.0f / static_cast<float>(dim.x);
const float pixel_half_height = 1.0f / static_cast<float>(dim.y);
```

### Step 4: Remove all Category A and B constants from OptiXData.h

In `namespace RenderingConstants`, remove these lines:
```cpp
constexpr float COLOR_BLACK = 0.0f;
constexpr float COLOR_WHITE = 1.0f;
constexpr float UNIT_CONVERSION_FACTOR = 1.0f;
constexpr float VACUUM_IOR = 1.0f;
constexpr float FRESNEL_BASE = 1.0f;
constexpr float METALLIC_THRESHOLD = 0.0f;
constexpr float DISTANCE_FALLOFF_NONE = 1.0f;
constexpr float DISTANCE_FALLOFF_BASE = 1.0f;
constexpr float DOT_PRODUCT_ZERO_THRESHOLD = 0.0f;
constexpr float DOT_PRODUCT_CLAMP_MIN = 0.0f;
constexpr float DOT_PRODUCT_CLAMP_MIN_SINGLE = 0.0f;
constexpr float FRESNEL_ONE_MINUS_R0 = 1.0f;
constexpr float FRESNEL_ONE_MINUS_COS = 1.0f;
```

Also remove `REFLECTION_SCALE` only if unused — but it IS used (`helpers.cu:662-664`), so **keep it**.

### Step 5: Verify no remaining references to removed constants

```bash
grep -rn "COLOR_BLACK\|COLOR_WHITE\|UNIT_CONVERSION_FACTOR\|FRESNEL_BASE\|METALLIC_THRESHOLD\|DISTANCE_FALLOFF_NONE\|DISTANCE_FALLOFF_BASE\|DOT_PRODUCT_ZERO_THRESHOLD\|DOT_PRODUCT_CLAMP_MIN\|FRESNEL_ONE_MINUS_R0\|FRESNEL_ONE_MINUS_COS\|VACUUM_IOR" optix-jni/src/main/native/
```
Expected: no matches in shader or header files.

### Step 6: Compile

```bash
sbt compile
```
Expected: clean compile (this exercises the CUDA shader compilation).

### Step 7: Run full test suite

```bash
sbt test
```
Expected: all tests pass with no regressions.

### Step 8: Close item in CODE_IMPROVEMENTS.md

### Step 9: Commit

```bash
git add optix-jni/src/main/native/include/OptiXData.h optix-jni/src/main/native/shaders/helpers.cu optix-jni/src/main/native/shaders/raygen_primary.cu CODE_IMPROVEMENTS.md
git commit -m "refactor: remove overly literal constants from RenderingConstants (M-naming-constants)

Removed 13 constants that were just restatements of 0.0f or 1.0f with names that
added indirection without encoding domain meaning (COLOR_BLACK, FRESNEL_BASE,
VACUUM_IOR, DISTANCE_FALLOFF_NONE, FRESNEL_ONE_MINUS_R0, etc.).
Replaced usages with literal values or MaterialConstants::IOR_VACUUM where meaningful."
```

---

## Final Verification

After all tasks are complete:

```bash
sbt test
```

Expected: all tests pass. No regressions introduced.

Confirm CODE_IMPROVEMENTS.md has no remaining Medium Priority items (all addressed or closed).
