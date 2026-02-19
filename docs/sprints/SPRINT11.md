# Sprint 11: libGDX Wrapper, Thin-Film Physics, and 4D Framework Enhancements

**Sprint:** 11
**Status:** Not Started
**Estimate:** 17-22 hours
**Branch:** `feature/sprint-11`
**Dependencies:** Sprint 10 (Scala DSL) - optional but recommended

---

## Goal

Three areas of work this sprint:

1. **libGDX Scala wrapper** — eliminate `var` and `null` from Scala code by delegating those concerns to a dedicated wrapper layer around libGDX.
2. **Thin-film physics** — implement proper thin-film interference for the Film material, with a thickness parameter and wavelength-dependent interference effects.
3. **4D framework enhancements** — complete the 4D user-experience features: interactive projection adjustment, view reset, CLI shortcuts, and state persistence.

## Success Criteria

- [ ] libGDX API calls in Scala use a wrapper layer; no `var` or `null` in non-wrapper Scala code related to libGDX
- [ ] Film material supports a `thickness` parameter (in nanometers) and renders visible interference fringes
- [ ] Interactive 4D projection adjustment (Shift+Scroll changes `eyeW`)
- [ ] Reset 4D view to defaults (ESC key)
- [ ] CLI: `--4d-rotation=XW,YW,ZW` shorthand for 4D rotation angles
- [ ] CLI: `--4d-preset=NAME` for common 4D views (edge-on, face-on, cell-on)
- [ ] State persistence: save/load 4D view parameters
- [ ] All tests pass (~20-30 new tests)

---

## Background

### Already Implemented (Sprint 8-10)

| Feature | Status |
|---------|--------|
| 4D math (`Rotation`, `Projection`, `Vector[4]`, `Matrix[4]`) | ✅ Complete |
| 4D objects (`Tesseract`, `TesseractSponge`, `TesseractSponge2`) | ✅ Complete |
| LibGDX 4D interaction (Shift+Arrow/Mouse) | ✅ Complete |
| OptiX 4D rendering (`Mesh4DProjection`) | ✅ Complete |
| OptiX 4D interaction (Shift+Arrow keys) | ✅ Complete (v0.4.2) |
| OptiX 4D mouse rotation (Shift+Drag) | ✅ Complete (v0.4.2) |
| Abstract `Mesh4D` interface | ✅ Complete |
| Film material (basic) | ✅ Complete (v0.5.0) |

### Remaining Features (This Sprint)

| Feature | Status | Priority |
|---------|--------|----------|
| Scala libGDX wrapper (no var/null) | ❌ Not implemented | High |
| Thin-film interference physics | ❌ Not implemented | High |
| Shift+Scroll projection adjustment | ❌ Not implemented | High |
| ESC to reset 4D view | ❌ Not implemented | High |
| CLI shortcuts (--4d-rotation, --4d-preset) | ❌ Not implemented | High |
| State persistence | ❌ Not implemented | Medium |

### Deferred to Sprint 12

| Feature | Reason |
|---------|--------|
| Per-instance 4D parameters | Stretch goal, deferred from this sprint |

---

## Tasks

### Task 11.1: Scala Wrapper for libGDX

**Estimate:** 4-5 hours

Introduce a thin Scala wrapper layer around libGDX to isolate all `var` and `null` usage. The goal is that all Scala code outside the wrapper interacts with idiomatic Scala (immutable values, `Option` instead of `null`, `val` instead of `var`), while the wrapper handles the mutable/nullable libGDX API internally.

#### Scope

- Identify all sites in the Scala codebase where libGDX types are held in `var`s or where `null` is used for libGDX values (e.g. `ApplicationListener`, `SpriteBatch`, `BitmapFont`, libGDX asset handles)
- Introduce wrapper objects / classes in a dedicated package (e.g. `menger.gdx`) that own the mutable state
- Expose immutable or `Option`-typed interfaces to callers
- Ensure no functional regressions (all existing tests pass)

#### Design Notes

- The wrapper layer is the *only* place that holds `var`s for libGDX objects; everything else accesses them through the wrapper's stable interface
- `null` returns from libGDX APIs (unloaded assets, missing fonts, etc.) become `Option[T]` at the wrapper boundary
- Lifecycle (`create()`, `dispose()`, etc.) remains internal to the wrapper

#### Tests to Add

- Unit tests verifying that wrapper methods return `Option` (not `null`) for absent resources
- Verify that previously `var`-backed state is correctly initialised and disposed through the wrapper lifecycle

---

### Task 11.2: Thin-Film Physics (Film Material)

**Estimate:** 5-6 hours

Replace the current Film material's approximate color tinting with a physically-based thin-film interference model. The interference color depends on the film thickness (in nanometers) and the angle of incidence, producing the characteristic rainbow iridescence seen in soap bubbles, oil slicks, and anti-reflective coatings.

#### Physics Background

Thin-film interference occurs when light reflects off the top and bottom surfaces of a thin transparent layer. The path difference is `2 * n * d * cos(θ_t)` where `n` is the film's refractive index, `d` is the thickness, and `θ_t` is the transmitted angle. Constructive interference for wavelength `λ` occurs when the path difference equals `m * λ`.

#### Files to Modify

- **`optix-jni/src/main/native/shaders/` (hit shader)** — add thin-film interference calculation; sample across visible spectrum (≈400–700 nm) and convert to RGB
- **`optix-jni/src/main/native/include/MaterialPresets.h`** — add `filmThicknessNm` field to the Film preset; default ~500 nm (green interference)
- **`optix-jni/src/main/native/include/OptiXData.h`** — add `filmThickness` field to the material struct
- **JNI bindings** — expose `setFilmThickness(float nm)` on `OptiXRenderer`
- **`menger-common/`** — add `filmThickness` to `Material` / `OptixMaterial`
- **`menger-app/` CLI** — add `--film-thickness=NM` option

#### Tests to Add

- Render Film sphere at several thickness values (200 nm, 400 nm, 600 nm) and assert that the dominant hue shifts with thickness
- Verify that at 0 nm thickness the film renders like a plain dielectric

---

### Task 11.3: Add Shift+Scroll for 4D Projection Adjustment

**Estimate:** 1.5 hours

Implement Shift+Scroll to adjust `eyeW` parameter for 4D projection distance.

#### Files to Modify

**`menger-app/src/main/scala/menger/input/OptiXCameraHandler.scala`**

Add projection adjustment handling:

```scala
override def handleScrolled(amountX: Float, amountY: Float, modifiers: ModifierState): Boolean =
  if modifiers.shift then
    handle4DProjectionScroll(amountY)
    true
  else
    handleZoom(amountY)
    true

private def handle4DProjectionScroll(amountY: Float): Unit =
  // Exponential scaling for smooth feel
  val factor = Math.pow(1.1, amountY.toDouble).toFloat
  val currentEyeW = getCurrentEyeW()  // Get from current 4D state
  val newEyeW = (currentEyeW * factor).max(getCurrentScreenW() + 0.1f)

  logger.debug(s"4D projection scroll: eyeW $currentEyeW -> $newEyeW")
  dispatcher.notifyObservers(ProjectionChangeEvent(newEyeW))
```

#### Tests to Add

```scala
class OptiXCameraHandler4DProjectionSpec extends AnyFlatSpec:
  it should "adjust eyeW on Shift+Scroll" in:
    // Test that Shift+Scroll modifies eyeW parameter

  it should "clamp eyeW above screenW" in:
    // Test that eyeW never goes below screenW + epsilon
```

---

### Task 11.4: Add ESC to Reset 4D View

**Estimate:** 1 hour

Add ESC key handler to reset 4D view to default parameters while keeping 3D camera unchanged.

#### Files to Modify

**`menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala`**

Modify ESC handling:

```scala
override protected def handleKeyPress(key: Key, modifiers: ModifierState): Boolean =
  key match
    case Key.Escape =>
      // Reset 4D view to defaults
      dispatcher.notifyObservers(Reset4DViewEvent())
      // Then exit
      if Gdx.app != null then Gdx.app.exit()
      true
```

---

### Task 11.5: Add CLI Shortcuts (--4d-rotation)

**Estimate:** 2.5 hours

Add convenient CLI options for specifying 4D parameters.

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Add new options:

```scala
val fourDRotation: ScallopOption[String] = opt[String](
  name = "4d-rotation", required = false,
  descr = "4D rotation angles as XW,YW,ZW in degrees (e.g., --4d-rotation=30,20,0)"
)
```

---

### Task 11.6: Documentation Updates

**Estimate:** 1 hour

Update documentation to reflect new features.

#### Files to Modify

**`CHANGELOG.md`**

```markdown
## [Unreleased]

### Added
- Scala wrapper layer for libGDX (no var/null outside wrapper)
- Thin-film interference physics for Film material (`--film-thickness=NM`)
- **4D Framework Enhancements**
  - Shift+Scroll to adjust 4D projection distance (eyeW)
  - ESC to reset 4D view to defaults
  - CLI shortcuts: `--4d-rotation=XW,YW,ZW` for quick rotation setup
  - CLI presets: `--4d-preset=NAME` (classic, edge-on, face-on, cell-on, flat)
  - State persistence: `--save-4d-state=FILE`, `--load-4d-state=FILE`
```

**`USER_GUIDE.md`**

Add section on thin-film thickness and 4D view presets.

**`TODO.md`**

Move completed items and update sprint 12 deferred list.

---

## Summary

| Task      | Description | Estimate | Priority |
|-----------|-------------|----------|----------|
| 11.1      | Scala libGDX wrapper | 4-5h | High |
| 11.2      | Thin-film physics | 5-6h | High |
| 11.3      | Shift+Scroll projection | 1.5h | High |
| 11.4      | ESC reset 4D view | 1h | High |
| 11.5      | CLI shortcuts | 2.5h | High |
| 11.6      | Documentation | 1h | High |
| **Total** | | **17-19h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated with new features
- [ ] Manual verification of thin-film rendering and 4D presets
- [ ] TODO.md updated

---

## Notes

### Implementation Order

Recommended order for minimal dependencies:

1. **Task 11.1** (libGDX wrapper) — foundational refactor, no feature dependencies
2. **Task 11.2** (Thin-film physics) — independent, C++/shader work
3. **Task 11.5** (CLI shortcuts) — foundation for 4D enhancements
4. **Task 11.3** (Shift+Scroll) — builds on existing input handling
5. **Task 11.4** (ESC reset) — simple extension
7. **Task 11.6** (Documentation) — last

### Testing Strategy

- **Unit tests**: libGDX wrapper boundary, FourDPresets, FourDState serialisation, thin-film colour at known thickness values
- **Integration tests**: CLI option parsing and validation, Film material renders
- **Manual tests**: Interactive Shift+Scroll and ESC; visual check of thin-film iridescence

### Deferred

- **Per-instance 4D parameters** (was 11.5): deferred to Sprint 12. Instances with identical 4D params share a mesh; different params require separate meshes — group specs by `(objectType, level, rotXW, rotYW, rotZW)`.

---

## References

- Sprint 8 (4D Foundation): `docs/arc42/adr/ADR-008-4d-projection.md`
- Sprint 9 (TesseractSponge): `CHANGELOG.md` v0.4.3
- Existing 4D input handling: `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala`
- Film material: `optix-jni/src/main/native/include/MaterialPresets.h`
