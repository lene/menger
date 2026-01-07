# OptiXEngine Constructor Refactoring Plan

**Issue:** CODE_IMPROVEMENTS.md #44 (Critical)
**Status:** COMPLETED
**Effort:** 4-6 hours (actual: ~2 hours)
**Date:** 2026-01-05

## Problem Statement

The `OptiXEngine` constructor has 22 parameters, violating the Single Responsibility Principle
and making the code difficult to maintain, test, and extend.

```scala
class OptiXEngine(
  val spongeType: String,
  val spongeLevel: Float,
  val lines: Boolean,                    // UNUSED in OptiX - remove
  val color: Color,
  val fpsLogIntervalMs: Int,
  val sphereRadius: Float,
  val ior: Float,
  val scale: Float,
  val cameraPos: Vector3,
  val cameraLookat: Vector3,
  val cameraUp: Vector3,
  val center: Vector3,
  val planeSpec: PlaneSpec,
  val planeColor: Option[PlaneColorSpec] = None,
  val timeout: Float = 0f,
  saveName: Option[String] = None,
  val enableStats: Boolean = false,
  val lights: Option[List[menger.LightSpec]] = None,  // Option[List] -> List
  val renderConfig: RenderConfig = RenderConfig.Default,
  val causticsConfig: CausticsConfig = CausticsConfig.Disabled,
  val maxInstances: Int = 64,
  val objectSpecs: Option[List[ObjectSpec]] = None
)(using profilingConfig: ProfilingConfig)
```

**Issues:**
- 22 parameters is excessive and error-prone
- Parameters from different concerns are mixed together
- Hard to remember parameter order
- Test setup is verbose and repetitive
- Adding new parameters makes the problem worse
- `lines` parameter is unused (OptiX doesn't support wireframe)
- `Option[List[LightSpec]]` is redundant - empty list works fine

---

## Discovery: Config Classes Already Exist!

The `menger.config` package already contains the grouped config classes:

| File | Class | Status |
|------|-------|--------|
| `CameraConfig.scala` | `CameraConfig(position, lookAt, up)` | Ready |
| `EnvironmentConfig.scala` | `EnvironmentConfig(plane, planeColor, lights)` | Needs fix: `lights` should be `List` not `Option[List]` |
| `ExecutionConfig.scala` | `ExecutionConfig(fpsLogIntervalMs, timeout, saveName, enableStats, maxInstances)` | Ready |
| `SceneConfig.scala` | `SceneConfig(spongeType, level, lines, color, sphereRadius, ior, scale, center, objectSpecs)` | Needs fix: remove `lines`, extract `MaterialConfig` |
| `OptiXEngineConfig.scala` | `OptiXEngineConfig(scene, camera, environment, execution, render, caustics)` | Ready |

**The config classes exist but OptiXEngine doesn't use them!**

---

## Solution: Use Existing Config + Minor Fixes

### Changes Required

#### 1. Fix `EnvironmentConfig`: `Option[List]` → `List`

**Before:**
```scala
case class EnvironmentConfig(
  plane: PlaneSpec,
  planeColor: Option[PlaneColorSpec] = None,
  lights: Option[List[LightSpec]] = None  // Unnecessary Option wrapper
)
```

**After:**
```scala
case class EnvironmentConfig(
  plane: PlaneSpec,
  planeColor: Option[PlaneColorSpec] = None,
  lights: List[LightSpec] = List.empty  // Empty list = no lights
)
```

#### 2. Add `MaterialConfig` and update `SceneConfig`

**New class:**
```scala
case class MaterialConfig(
  color: Color = Color.WHITE,
  ior: Float = 1.5f
)

object MaterialConfig:
  val Default: MaterialConfig = MaterialConfig()
  val Glass: MaterialConfig = MaterialConfig(color = Color(1f, 1f, 1f, 0.1f), ior = 1.5f)
  val Diamond: MaterialConfig = MaterialConfig(color = Color.WHITE, ior = 2.42f)
```

**Updated SceneConfig:**
```scala
case class SceneConfig(
  spongeType: String = "sphere",
  level: Float = 0f,
  // lines: Boolean = false,  // REMOVED - not used in OptiX
  material: MaterialConfig = MaterialConfig.Default,
  sphereRadius: Float = 0.5f,
  scale: Float = 1.0f,
  center: Vector3 = Vector3.Zero,
  objectSpecs: Option[List[ObjectSpec]] = None
)
```

#### 3. Update `OptiXEngine` to use `OptiXEngineConfig`

**New constructor:**
```scala
class OptiXEngine(config: OptiXEngineConfig)(using profilingConfig: ProfilingConfig)
```

**Result:** 1 top-level parameter (down from 22)

---

## Migration Strategy

### Phase 1: Fix Config Classes

1. Update `EnvironmentConfig`: change `lights: Option[List[LightSpec]]` to `lights: List[LightSpec]`
2. Create `MaterialConfig` case class
3. Update `SceneConfig`: remove `lines`, replace `color`/`ior` with `material: MaterialConfig`

### Phase 2: Update OptiXEngine

1. Replace 22-parameter constructor with single `OptiXEngineConfig` parameter
2. Update internal code to read from config object
3. Remove old constructor entirely (no deprecation)

### Phase 3: Update Call Sites

1. Update `Main.scala` to build `OptiXEngineConfig` and pass to constructor
2. Update `OptiXEngineSuite.scala` tests

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/main/scala/menger/config/MaterialConfig.scala` | **New file** |
| `src/main/scala/menger/config/EnvironmentConfig.scala` | `lights: Option[List]` → `lights: List` |
| `src/main/scala/menger/config/SceneConfig.scala` | Remove `lines`, add `material: MaterialConfig` |
| `src/main/scala/menger/engines/OptiXEngine.scala` | Replace constructor with `(config: OptiXEngineConfig)` |
| `src/main/scala/Main.scala` | Build `OptiXEngineConfig` in `createOptiXEngine` |
| `src/test/scala/menger/OptiXEngineSuite.scala` | Update tests to use new API |

---

## Usage Examples

### Before (22 parameters)

```scala
OptiXEngine(
  opts.objectType.toOption.getOrElse("sphere"),
  opts.level(), opts.lines(), opts.color(),
  opts.fpsLogInterval(),
  opts.radius(), opts.ior(), opts.scale(),
  opts.cameraPos(), opts.cameraLookat(), opts.cameraUp(), 
  opts.center(), opts.plane(),
  opts.planeColor.toOption,
  opts.timeout(),
  opts.saveName.toOption,
  opts.stats(),
  opts.light.toOption,
  opts.renderConfig,
  opts.causticsConfig,
  opts.maxInstances(),
  opts.objects.toOption
)
```

### After (Single config object)

```scala
OptiXEngine(OptiXEngineConfig(
  scene = SceneConfig(
    spongeType = opts.objectType.toOption.getOrElse("sphere"),
    level = opts.level(),
    material = MaterialConfig(opts.color(), opts.ior()),
    sphereRadius = opts.radius(),
    scale = opts.scale(),
    center = opts.center(),
    objectSpecs = opts.objects.toOption
  ),
  camera = CameraConfig(opts.cameraPos(), opts.cameraLookat(), opts.cameraUp()),
  environment = EnvironmentConfig(
    plane = opts.plane(),
    planeColor = opts.planeColor.toOption,
    lights = opts.light.toOption.getOrElse(List.empty)
  ),
  execution = ExecutionConfig(
    fpsLogIntervalMs = opts.fpsLogInterval(),
    timeout = opts.timeout(),
    saveName = opts.saveName.toOption,
    enableStats = opts.stats(),
    maxInstances = opts.maxInstances()
  ),
  render = opts.renderConfig,
  caustics = opts.causticsConfig
))
```

### Test Setup (Before)

```scala
new OptiXEngine(
  spongeType = "sphere",
  spongeLevel = 0f,
  lines = false,
  color = Color.WHITE,
  fpsLogIntervalMs = 1000,
  sphereRadius = radius,
  ior = ior,
  scale = scale,
  cameraPos = Vector3(0f, 0.5f, 3.0f),
  cameraLookat = Vector3(0f, 0f, 0f),
  cameraUp = Vector3(0f, 1f, 0f),
  center = Vector3(0f, 0f, 0f),
  planeSpec = PlaneSpec(Axis.Y, false, -2f),
  timeout = timeout,
  saveName = None,
  enableStats = enableStats,
  renderConfig = renderConfig
)
```

### Test Setup (After)

```scala
OptiXEngine(OptiXEngineConfig(
  scene = SceneConfig(
    spongeType = "sphere",
    material = MaterialConfig(Color.WHITE, ior),
    sphereRadius = radius
  ),
  camera = CameraConfig.Default,
  environment = EnvironmentConfig.WithPlane,
  execution = ExecutionConfig(timeout = timeout, enableStats = enableStats),
  render = renderConfig
))
```

---

## Design Decisions

### D1: Use existing config classes

**Decision:** Reuse the existing `menger.config` package classes.

**Rationale:** 
- Config classes already exist and are well-designed
- No need to create duplicate structures
- Just need minor fixes and to wire them to OptiXEngine

### D2: `lights: List` instead of `Option[List]`

**Decision:** Change `lights: Option[List[LightSpec]]` to `lights: List[LightSpec] = List.empty`.

**Rationale:**
- `None` and `Some(List())` are semantically equivalent
- Eliminates unnecessary `.getOrElse(List())` calls
- More idiomatic Scala

### D3: Remove `lines` parameter

**Decision:** Remove `lines` from `SceneConfig` and `OptiXEngine`.

**Rationale:**
- OptiX renders ray-traced solid geometry only
- Wireframe (`GL_LINES`) is a LibGDX/OpenGL feature
- Parameter was passed but never used in OptiX code

### D4: Extract `MaterialConfig`

**Decision:** Create `MaterialConfig(color, ior)` and use in `SceneConfig`.

**Rationale:**
- `color` and `ior` are material properties (surface appearance)
- Logical grouping separate from geometry (sphereRadius, center, scale)
- Enables material presets (Glass, Diamond, etc.)

### D5: Remove old constructor entirely

**Decision:** Replace 22-param constructor with `(config: OptiXEngineConfig)`, no deprecation.

**Rationale:**
- No external consumers depend on the old API
- Cleaner codebase without deprecated methods
- One-time migration is manageable

---

## Validation Checklist

After implementation:

- [x] All 897+ tests pass (939 tests confirmed passing)
- [x] `sbt compile` succeeds without warnings
- [x] `EnvironmentConfig.lights` is `List` not `Option[List]`
- [x] `MaterialConfig` exists with `Default`, `Glass`, `Diamond`, `Mirror`, `Water` presets
- [x] `SceneConfig` has `material: MaterialConfig`, no `lines`, no `color`/`ior`
- [x] `OptiXEngine` constructor takes single `OptiXEngineConfig` parameter
- [x] `Main.scala` builds config and passes to constructor
- [x] `OptiXEngineSuite` tests use new API
- [x] `MainSuite` tests use new API (with `sphereRadius` accessor)
- [x] No functionality changes - pure refactoring

---

## Open Questions (Answered)

1. **Why is `lights` an `Option[List]`?**
   - **Answer:** No good reason. Change to `List` with empty default.

2. **Where does `lines` come from if unused?**
   - **Answer:** CLI flag for LibGDX wireframe rendering. Not applicable to OptiX. Remove.

3. **Should `color` and `ior` be in MaterialConfig?**
   - **Answer:** Yes, they are material properties. Create `MaterialConfig`.

4. **Deprecate or remove old API?**
   - **Answer:** Remove entirely. No external consumers.

---

## Completion Notes (2026-01-05)

### Files Modified

| File | Change |
|------|--------|
| `src/main/scala/menger/config/MaterialConfig.scala` | **NEW** - Material properties with presets |
| `src/main/scala/menger/config/EnvironmentConfig.scala` | `lights: Option[List]` → `List` |
| `src/main/scala/menger/config/SceneConfig.scala` | Removed `lines`, added `material: MaterialConfig` |
| `src/main/scala/menger/engines/OptiXEngine.scala` | New single-param constructor |
| `src/main/scala/menger/optix/SceneConfigurator.scala` | Updated for `List[LightSpec]` |
| `src/main/scala/Main.scala` | Build `OptiXEngineConfig` and pass to constructor |
| `src/test/scala/menger/OptiXEngineSuite.scala` | Use new config-based API |
| `src/test/scala/MainSuite.scala` | Use new config-based API (via `sphereRadius` accessor) |

### Additional Implementation Notes

1. Added `sphereRadius` accessor to `OptiXEngine` to maintain test compatibility
2. Added `timeout` override to satisfy `TimeoutSupport` trait requirement
3. Fixed import organization in `SceneConfig.scala` per scalafix rules
4. `MaterialConfig` includes 5 presets: `Default`, `Glass`, `Diamond`, `Mirror`, `Water`

### Verification

- All 939 Scala tests pass
- All 27 C++ tests pass
- Scalafix check passes
- Pre-push hook passes (including GPU/host memory leak checks)
