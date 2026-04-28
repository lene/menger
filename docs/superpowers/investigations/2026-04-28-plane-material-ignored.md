# Investigation: `--plane-material chrome/gold` has no visual effect

**Opened:** 2026-04-28
**Status:** Resolved

---

## Description (v-final)

`--plane-material` has no effect: all three variants (`chrome`, `gold`, default) produce
identical pixel output regardless of material. Expected: plane exhibits material's appearance
(reflective for chrome/gold, matte for default).

---

## Stage 0 — Context

**Command:** `./menger-app/target/universal/stage/bin/menger-app -o --headless -s /tmp/a.png --objects type=sphere --plane-material chrome`

Three variants rendered and MD5-compared:
- `--plane-material chrome`
- `--plane-material gold`
- (no flag)

All three: identical MD5 → confirmed symptom.

**Pipeline stages identified:**
- CLI parsing: `MengerCLIOptions.scala`, `Main.scala`
- PlaneConfig construction: `Main.scala:175-179`
- SceneConfigurator routing: `SceneConfigurator.configurePlanes()`
- JNI layer: `OptiXRenderer.addPlaneSolidColorWithMaterial`
- Native: `RenderConfig.addPlaneSolidColorWithMaterial`, `RenderConfig.cpp`
- Shader: `miss_plane.cu` `__miss__ms()`

---

## Stage 1 — Detector

**Command:** `sbt "optixJni/testOnly menger.optix.PlaneMaterialSuite"`

**Output:** 8/8 pass — including "plane material chrome vs matte should render differently".

`PlaneMaterialSuite` calls JNI directly (bypasses CLI). Chrome and matte DO produce different
renders at the JNI level when the scene has a correctly positioned light. This proves the
rendering engine supports material effects for planes; the bug is upstream in the CLI→JNI path.

**Detector validated:** passes on known-good (JNI direct call with chrome material) and was
confirmed to catch bugs (CLI path produces identical renders despite different material flags).

---

## Stage 2 — Localization

**Command:** `--plane-color cccccc --plane-material chrome` vs `--plane-color cccccc --plane-material matte`

Both produced identical MD5 even with `--plane-color`. Investigation of `SceneConfigurator.scala:62-95`:

- `colorSpec = Some` branches (solid + checker) correctly check `planeConfig.material` and call
  `addPlaneSolidColorWithMaterial` / `addPlaneCheckerColorsWithMaterial` when material is set.
- `colorSpec = None` branch (line 93-95): always calls `renderer.addPlane()`, material ignored.

Also examined `miss_plane.cu` `__miss__ms()`: plane lighting uses Phong-only model (diffuse +
Phong specular). Chrome has `roughness=0.0` → `spec_power = 2000` → pinpoint specular lobe.
Default camera/light angle places the specular spot off-screen; chrome and matte appear identical
even when material IS passed through. This is a second root cause: plane shading is Phong-only,
not PBR — metallic planes cannot show reflections.

**Evidence table:**

| Path | Material applied? | Visible in default scene? |
|------|------------------|--------------------------|
| `--plane-material chrome` (no `--plane-color`) | ❌ dropped at `SceneConfigurator:93` | N/A |
| `--plane-material chrome --plane-color cccccc` | ✅ reaches `addPlaneSolidColorWithMaterial` | ❌ Phong specular off-screen |
| JNI direct (`PlaneMaterialSuite`) | ✅ | ✅ test scene has correct light angle |

**Prime suspects confirmed:**
1. `SceneConfigurator.configurePlanes()` `case None` branch silently drops material
2. `miss_plane.cu` Phong-only model — no reflected rays for metallic planes

---

## Stage 3 — Root Cause and Fix

### Bug 1 — `SceneConfigurator.scala:93-95`

`case None` branch always calls `renderer.addPlane()` regardless of `planeConfig.material`.

**Fix:** Check `planeConfig.material` in the `case None` branch. When material is set, call
`addPlaneSolidColorWithMaterial(mat.color, mat)` — using the material's own color as a solid
floor. The checker pattern is opt-in via `--plane-color RRGGBB:RRGGBB` only. When material
is None, keep calling `addPlane` (default checker gray, unchanged behavior).

File: `menger-app/src/main/scala/menger/optix/SceneConfigurator.scala:93-101`

### Bug 2 — `miss_plane.cu` metallic path

Miss shader computed `total_color = base_color * lighting + Phong_specular`. No secondary
rays were traced. Metallic planes could not reflect scene geometry.

**Fix:** Add metallic branch in `__miss__ms()`:
- Read depth from `optixGetPayload_3()`
- Call `traceReflectedRay()` (already in `helpers.cu`, which is included before `miss_plane.cu`)
- Tint reflected color by plane base color (for gold/copper tint)
- Blend: `metallic * reflected + (1 - metallic) * diffuse`
- Non-metallic planes keep the existing Phong path

File: `optix-jni/src/main/native/shaders/miss_plane.cu`

### Regression tests added

`PlaneMaterialSuite`:
- `"default-gray chrome checker plane should render differently from plain default plane"` — verifies Bug 1 fix path at JNI level
- `"metallic plane should reflect scene geometry"` — verifies Bug 2 fix: chrome plane with geometry above produces different render from matte plane (diff > 1.0 mean pixel error)

### Visual crosscheck

Four CLI renders compared before/after fix:

| Render | Before fix | After fix |
|--------|-----------|-----------|
| chrome no-color vs matte no-color | identical MD5 | **different** (floor reflects background) |
| chrome with-color vs matte with-color | identical MD5 | **different** (floor reflects background) |

Chrome floor with default camera shows purple tint (reflecting dark background sky) vs
gray matte floor. Confirmed: correct artifacts changed, no unexpected regressions.

### Test results

1861/1861 pass after fix.

---

## Resolution

**Bug 1:** Fixed in `SceneConfigurator.scala` — `case None` branch now applies material when set.

**Bug 2:** Fixed in `miss_plane.cu` — metallic planes trace reflected rays via `traceReflectedRay()`,
matching sphere `handleMetallicOpaque()` behavior.

The plane shading model now matches the sphere shading model for metallic materials.
