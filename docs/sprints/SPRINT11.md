# Sprint 11: 4D Framework Enhancements

**Sprint:** 11 - 4D Framework Completion
**Status:** Not Started
**Estimate:** 8-10 hours
**Branch:** `feature/sprint-11`
**Dependencies:** Sprint 10 (Scala DSL) - optional but recommended

---

## Goal

Complete the 4D manipulation framework with convenience features: projection adjustment, view presets, state persistence, and per-instance 4D parameters. This sprint completes the user experience features that make 4D exploration more intuitive.

## Success Criteria

- [ ] Interactive 4D projection adjustment (Shift+Scroll changes `eyeW`)
- [ ] Reset 4D view to defaults (ESC key)
- [ ] CLI: `--4d-rotation=XW,YW,ZW` shorthand for 4D rotation angles
- [ ] CLI: `--4d-preset=NAME` for common 4D views (edge-on, face-on, cell-on)
- [ ] State persistence: save/load 4D view parameters
- [ ] Per-instance 4D parameters in multi-object scenes (optional stretch goal)
- [ ] All tests pass (~15-20 new tests)

---

## Background

### Already Implemented (Sprint 8-9)

| Feature | Status |
|---------|--------|
| 4D math (`Rotation`, `Projection`, `Vector[4]`, `Matrix[4]`) | ✅ Complete |
| 4D objects (`Tesseract`, `TesseractSponge`, `TesseractSponge2`) | ✅ Complete |
| LibGDX 4D interaction (Shift+Arrow/Mouse) | ✅ Complete |
| OptiX 4D rendering (`Mesh4DProjection`) | ✅ Complete |
| OptiX 4D interaction (Shift+Arrow keys) | ✅ Complete (v0.4.2) |
| OptiX 4D mouse rotation (Shift+Drag) | ✅ Complete (v0.4.2) |
| Abstract `Mesh4D` interface | ✅ Complete |

### Remaining Features (This Sprint)

| Feature | Status | Priority |
|---------|--------|----------|
| Shift+Scroll projection adjustment | ❌ Not implemented | High |
| ESC to reset 4D view | ❌ Not implemented | High |
| CLI shortcuts (--4d-rotation, --4d-preset) | ❌ Not implemented | High |
| State persistence | ❌ Not implemented | Medium |
| Per-instance 4D parameters | ❌ Not implemented | Low (stretch) |

---

## Tasks

### Task 11.1: Add Shift+Scroll for 4D Projection Adjustment

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

### Task 11.2: Add ESC to Reset 4D View

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

### Task 11.3: Add CLI Shortcuts (--4d-rotation, --4d-preset)

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

val fourDPreset: ScallopOption[String] = opt[String](
  name = "4d-preset", required = false,
  descr = "4D view preset: classic (default), edge-on, face-on, cell-on, flat"
)

// Validation: cannot specify both
validateOpt(fourDRotation, rotXW, rotYW, rotZW) { (rotation, xw, yw, zw) =>
  if rotation.isDefined && (xw.isDefined || yw.isDefined || zw.isDefined) then
    Left("Cannot specify both --4d-rotation and individual --rot-xw/--rot-yw/--rot-zw")
  else
    Right(())
}
```

#### Files to Create

**`menger-app/src/main/scala/menger/config/FourDPresets.scala`**

```scala
package menger.config

import menger.RotationProjectionParameters
import menger.common.Const

object FourDPresets:
  val presets: Map[String, RotationProjectionParameters] = Map(
    "classic" -> RotationProjectionParameters(
      rotXW = 15f, rotYW = 10f, rotZW = 0f,
      eyeW = Const.defaultEyeW, screenW = Const.defaultScreenW
    ),
    "edge-on" -> RotationProjectionParameters(
      rotXW = 45f, rotYW = 0f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "face-on" -> RotationProjectionParameters(
      rotXW = 0f, rotYW = 0f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "cell-on" -> RotationProjectionParameters(
      rotXW = 45f, rotYW = 45f, rotZW = 0f,
      eyeW = 3f, screenW = 1.5f
    ),
    "flat" -> RotationProjectionParameters(
      rotXW = 0f, rotYW = 0f, rotZW = 0f,
      eyeW = 10f, screenW = 9f  // Nearly orthographic
    )
  )

  def get(name: String): Option[RotationProjectionParameters] =
    presets.get(name.toLowerCase)
```

#### Tests to Add

```scala
class FourDPresetsSpec extends AnyFlatSpec:
  it should "define all standard presets" in:
    FourDPresets.presets.keys should contain allOf ("classic", "edge-on", "face-on", "cell-on", "flat")

  it should "ensure eyeW > screenW for all presets" in:
    FourDPresets.presets.values.foreach { params =>
      params.eyeW should be > params.screenW
    }
```

---

### Task 11.4: State Persistence (Save/Load 4D View)

**Estimate:** 2 hours

Add ability to save and restore 4D view parameters.

#### Files to Create

**`menger-app/src/main/scala/menger/config/FourDState.scala`**

```scala
package menger.config

import scala.util.Try
import menger.RotationProjectionParameters

case class FourDState(
  rotXW: Float,
  rotYW: Float,
  rotZW: Float,
  eyeW: Float,
  screenW: Float
):
  def toParams: RotationProjectionParameters =
    RotationProjectionParameters(rotXW, rotYW, rotZW, eyeW, screenW)

object FourDState:
  def fromParams(params: RotationProjectionParameters): FourDState =
    FourDState(params.rotXW, params.rotYW, params.rotZW, params.eyeW, params.screenW)

  def toJson(state: FourDState): String =
    s"""{
       |  "rotXW": ${state.rotXW},
       |  "rotYW": ${state.rotYW},
       |  "rotZW": ${state.rotZW},
       |  "eyeW": ${state.eyeW},
       |  "screenW": ${state.screenW}
       |}""".stripMargin

  def fromJson(json: String): Try[FourDState] = Try:
    val rotXW = extractFloat(json, "rotXW")
    val rotYW = extractFloat(json, "rotYW")
    val rotZW = extractFloat(json, "rotZW")
    val eyeW = extractFloat(json, "eyeW")
    val screenW = extractFloat(json, "screenW")
    FourDState(rotXW, rotYW, rotZW, eyeW, screenW)

  private def extractFloat(json: String, key: String): Float =
    val pattern = s""""$key":\\s*([\\d.\\-]+)""".r
    pattern.findFirstMatchIn(json).map(_.group(1).toFloat).getOrElse(0f)

  def save(state: FourDState, path: String): Try[Unit] = Try:
    java.nio.file.Files.writeString(java.nio.file.Path.of(path), toJson(state))

  def load(path: String): Try[FourDState] =
    Try(java.nio.file.Files.readString(java.nio.file.Path.of(path))).flatMap(fromJson)
```

#### CLI Options

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

```scala
val save4DState: ScallopOption[String] = opt[String](
  name = "save-4d-state", required = false,
  descr = "Save current 4D view state to file on exit"
)

val load4DState: ScallopOption[String] = opt[String](
  name = "load-4d-state", required = false,
  descr = "Load 4D view state from file"
)
```

---

### Task 11.5: Per-Instance 4D Parameters (Stretch Goal)

**Estimate:** 3 hours

Enable different 4D rotations for different tesseract instances in the same scene.

**Note:** This is a stretch goal. If time is limited, defer to later sprint.

#### Architecture Decision

Instances with identical 4D params share a mesh. Different params require separate meshes.

#### Files to Modify

**`menger-app/src/main/scala/menger/engines/OptiXEngine.scala`**

Group specs by mesh identity (type + level + 4D params):

```scala
private case class MeshKey(
  objectType: String,
  level: Option[Float],
  rotXW: Float,
  rotYW: Float,
  rotZW: Float
)

private def groupSpecsByMeshIdentity(specs: List[ObjectSpec]): Map[MeshKey, List[ObjectSpec]] =
  specs.groupBy { spec =>
    MeshKey(spec.objectType, spec.level, spec.rotXW, spec.rotYW, spec.rotZW)
  }
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
- **4D Framework Enhancements (Sprint 11)**
  - Shift+Scroll to adjust 4D projection distance (eyeW)
  - ESC to reset 4D view to defaults
  - CLI shortcuts: `--4d-rotation=XW,YW,ZW` for quick rotation setup
  - CLI presets: `--4d-preset=NAME` (classic, edge-on, face-on, cell-on, flat)
  - State persistence: `--save-4d-state=FILE`, `--load-4d-state=FILE`
  - Per-instance 4D parameters for multi-tesseract scenes
```

**`USER_GUIDE.md`**

Add section on 4D view presets:

```markdown
### 4D View Presets

For quick setup, use predefined 4D views:

```bash
# Edge-on view (one edge points toward viewer)
menger --optix --objects type=tesseract --4d-preset=edge-on

# Face-on view (one face perpendicular to view)
menger --optix --objects type=tesseract --4d-preset=face-on

# Cell-on view (diagonal through hypercube)
menger --optix --objects type=tesseract --4d-preset=cell-on

# Flat projection (nearly orthographic)
menger --optix --objects type=tesseract --4d-preset=flat
```

Available presets: `classic`, `edge-on`, `face-on`, `cell-on`, `flat`
```

**`TODO.md`**

Move completed items:

```markdown
## Completed in Sprint 11
- [x] 4D projection adjustment (Shift+Scroll)
- [x] 4D view reset (ESC)
- [x] 4D CLI shortcuts
- [x] 4D view presets
- [x] 4D state persistence
```

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 11.1 | Shift+Scroll projection | 1.5h | High |
| 11.2 | ESC reset 4D view | 1h | High |
| 11.3 | CLI shortcuts & presets | 2.5h | High |
| 11.4 | State persistence | 2h | Medium |
| 11.5 | Per-instance 4D params | 3h | Low (stretch) |
| 11.6 | Documentation | 1h | High |
| **Total** | | **8-11h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated with new features
- [ ] Manual verification of 4D presets
- [ ] TODO.md updated

---

## Notes

### Implementation Order

Recommended order for minimal dependencies:

1. **Task 11.3** (CLI shortcuts) - Foundation, independent
2. **Task 11.1** (Shift+Scroll) - Builds on existing input handling
3. **Task 11.2** (ESC reset) - Simple extension
4. **Task 11.4** (State persistence) - Uses CLI from 11.3
5. **Task 11.5** (Per-instance) - Optional stretch goal
6. **Task 11.6** (Documentation) - Last

### Testing Strategy

- **Unit tests**: FourDPresets, FourDState serialization
- **Integration tests**: CLI option parsing and validation
- **Manual tests**: Interactive testing of Shift+Scroll and ESC

### Design Philosophy

This sprint focuses on **usability enhancements** rather than new core features. All the foundational 4D work was completed in Sprints 8-9. We're now adding quality-of-life features that make 4D exploration more intuitive and reproducible.

---

## References

- Original Sprint 10 plan (4D Framework): `docs/sprints/SPRINT10_OLD_BACKUP.md`
- Sprint 8 (4D Foundation): `docs/arc42/adr/ADR-008-4d-projection.md`
- Sprint 9 (TesseractSponge): `CHANGELOG.md` v0.4.3
- Existing 4D input handling: `menger-app/src/main/scala/menger/input/OptiXKeyHandler.scala`
