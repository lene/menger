# Create `menger-common` Module and Refactor to Vector[3]

## Overview
Create a new `menger-common` SBT subproject to share types between `menger` (root) and `optix-jni`. Move shared types to common module, then refactor coordinate arrays to use `Vector[3]`.

---

## Phase 1: Create `menger-common` Module Structure

### 1.1 Create Directory Structure
```
menger-common/
├── src/
│   ├── main/scala/menger/common/
│   │   ├── Vector.scala          (moved from menger/objects/)
│   │   ├── Const.scala           (moved from menger/)
│   │   ├── ImageSize.scala       (moved from optix-jni/)
│   │   ├── Light.scala           (unified from both projects)
│   │   └── Vec3.scala            (moved from menger/)
│   └── test/scala/menger/common/
│       └── VectorTest.scala      (if any tests exist)
```

### 1.2 Update `build.sbt`
Add new subproject and update dependencies:

```scala
// New subproject
lazy val mengerCommon = project
  .in(file("menger-common"))
  .settings(
    name := "menger-common",
    commonSettings,
    libraryDependencies ++= Seq(
      // Minimal dependencies - pure Scala only
      "org.scalatest" %% "scalatest" % scalaTestVersion % Test
    )
  )

// Update existing projects
lazy val optixJni = project
  .in(file("optix-jni"))
  .dependsOn(mengerCommon)  // ADD THIS
  .settings(...)

lazy val base = project
  .in(file("."))
  .dependsOn(optixJni, mengerCommon)  // ADD mengerCommon
  .settings(...)
```

---

## Phase 2: Move Core Types to Common Module

### 2.1 Move `Const.scala` (no dependencies)
- **From:** `src/main/scala/menger/Const.scala`
- **To:** `menger-common/src/main/scala/menger/common/Const.scala`
- **Package change:** `package menger` → `package menger.common`
- **Update imports in root:** `import menger.Const` → `import menger.common.Const`

### 2.2 Move `Vector.scala` (depends on Const)
- **From:** `src/main/scala/menger/objects/Vector.scala`
- **To:** `menger-common/src/main/scala/menger/common/Vector.scala`
- **Package change:** `package menger.objects` → `package menger.common`
- **Update internal dependency:** Use `Const.epsilon` from `menger.common.Const`
- **Update imports in root:** `import menger.objects.Vector` → `import menger.common.Vector`
- **Files affected:** ~60 import statements in main project (Matrix.scala, higher_d/*.scala, etc.)

### 2.3 Move `ImageSize` (no dependencies)
- **From:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` (lines 57-59)
- **To:** `menger-common/src/main/scala/menger/common/ImageSize.scala`
- **Package change:** `package menger.optix` → `package menger.common`
- **Update imports:**
  - In optix-jni: `import menger.common.ImageSize`
  - In root: `import menger.optix.ImageSize` → `import menger.common.ImageSize`

### 2.4 Move `Vec3` type alias
- **From:** `src/main/scala/menger/Vec3.scala`
- **To:** `menger-common/src/main/scala/menger/common/Vec3.scala`
- **Package change:** `package menger` → `package menger.common`
- **Update imports in root:** `import menger.Vec3` → `import menger.common.Vec3`

---

## Phase 3: Unify Light Types

### 3.1 Create Unified Light in Common Module
**File:** `menger-common/src/main/scala/menger/common/Light.scala`

```scala
package menger.common

enum LightType:
  case Directional, Point

sealed trait Light:
  def lightType: LightType
  def color: Vector[3]
  def intensity: Float

object Light:
  case class Directional(
    direction: Vector[3],
    color: Vector[3] = Vector[3](1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Directional

  case class Point(
    position: Vector[3],
    color: Vector[3] = Vector[3](1.0f, 1.0f, 1.0f),
    intensity: Float = 1.0f
  ) extends Light:
    val lightType = LightType.Point
```

### 3.2 Remove Old Light Types
- **Remove from optix-jni:** Lines 11-55 in `OptiXRenderer.scala` (LightType enum, Light case class, factory methods)
- **Remove from root:** Lines 288-291 in `MengerCLIOptions.scala` (LightType enum, LightSpec case class)

### 3.3 Update OptiXRenderer JNI Boundary
Keep Array[Float] at JNI boundary for C++ interop:

```scala
// Private helper for JNI conversion
private def lightToArrays(light: menger.common.Light): (Int, Array[Float], Array[Float], Array[Float], Float) =
  light match
    case Light.Directional(dir, color, intensity) =>
      (0, dir.toArray, Array(0f, 0f, 0f), color.toArray, intensity)
    case Light.Point(pos, color, intensity) =>
      (1, Array(0f, 0f, 0f), pos.toArray, color.toArray, intensity)

@native private def setLightsNative(
  types: Array[Int],
  directions: Array[Array[Float]],
  positions: Array[Array[Float]],
  colors: Array[Array[Float]],
  intensities: Array[Float],
  count: Int
): Unit

def setLights(lights: Seq[menger.common.Light]): Unit =
  // Convert and call native method
```

### 3.4 Update Root Project Light Usage
**File:** `src/main/scala/menger/OptiXResources.scala`

Remove conversion code (lines 86-100), use `menger.common.Light` directly:

```scala
import menger.common.Light

// LightSpec from CLI → menger.common.Light
def convertLight(spec: menger.LightSpec): Light =
  spec.lightType match
    case menger.LightType.DIRECTIONAL =>
      Light.Directional(
        direction = Vector[3](spec.position.x, spec.position.y, spec.position.z),
        color = Vector[3](spec.color.r, spec.color.g, spec.color.b),
        intensity = spec.intensity
      )
    case menger.LightType.POINT =>
      Light.Point(
        position = Vector[3](spec.position.x, spec.position.y, spec.position.z),
        color = Vector[3](spec.color.r, spec.color.g, spec.color.b),
        intensity = spec.intensity
      )
```

---

## Phase 4: Add Vector[3] Extension Methods

**File:** `menger-common/src/main/scala/menger/common/package.scala`

```scala
package menger.common

extension (v: Vector[3])
  def x: Float = v(0)
  def y: Float = v(1)
  def z: Float = v(2)
  def toArray: Array[Float] = Array(v(0), v(1), v(2))

object Vector3:
  def apply(x: Float, y: Float, z: Float): Vector[3] =
    Vector[3](x, y, z)

  def fromArray(arr: Array[Float]): Vector[3] =
    require(arr.length == 3, s"Expected 3 elements, got ${arr.length}")
    Vector[3](arr(0), arr(1), arr(2))
```

---

## Phase 5: Refactor OptiXRenderer to Vector[3]

### 5.1 Add Vector[3] Overloads
**File:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`

```scala
import menger.common.{Vector, Vector3}

// Existing native method (keep):
@native private def setCameraNative(
  eye: Array[Float],
  lookAt: Array[Float],
  up: Array[Float],
  fov: Float
): Unit

// New public API:
def setCamera(eye: Vector[3], lookAt: Vector[3], up: Vector[3], fov: Float): Unit =
  setCameraNative(eye.toArray, lookAt.toArray, up.toArray, fov)

// Mark old as deprecated:
@deprecated("Use setCamera(Vector[3], Vector[3], Vector[3], Float)", "0.4.0")
def setCamera(eye: Array[Float], lookAt: Array[Float], up: Array[Float], fov: Float): Unit =
  setCameraNative(eye, lookAt, up, fov)
```

Apply same pattern to:
- `setLight(direction: Array[Float], intensity: Float)` → deprecated, add Vector[3] version
- `setLights(lights: Seq[Light])` → already uses `menger.common.Light` with Vector[3]

---

## Phase 6: Update Call Sites Incrementally

### 6.1 Update TestScenarios
**File:** `optix-jni/src/test/scala/menger/optix/TestScenarios.scala`

```scala
import menger.common.{Vector, Vector3}

case class CameraConfig(
  eye: Vector[3] = Vector3(0.0f, 0.5f, 3.0f),
  lookAt: Vector[3] = Vector3(0.0f, 0.0f, 0.0f),
  up: Vector[3] = Vector3(0.0f, 1.0f, 0.0f),
  horizontalFov: Float = 60.0f
)

case class LightConfig(
  direction: Vector[3] = Vector3(0.5f, 0.5f, -0.5f),
  intensity: Float = 1.0f
)
```

### 6.2 Update Test Files (Incremental)
Let deprecation warnings guide migration:

```scala
import menger.common.Vector3

// Before:
val eye = Array(0.0f, 0.5f, 3.0f)

// After:
val eye = Vector3(0.0f, 0.5f, 3.0f)
```

**Files to update:** ~30 test files (can be done incrementally)

### 6.3 Update OptiXResources
**File:** `src/main/scala/menger/OptiXResources.scala`

```scala
import menger.common.Vector3

// Before:
val eye = Array(cameraPos.x, cameraPos.y, cameraPos.z)

// After:
val eye = Vector3(cameraPos.x, cameraPos.y, cameraPos.z)
```

---

## Phase 7: Cleanup (Future)

After all call sites migrated:
- Remove `@deprecated` Array[Float] methods
- Keep native JNI methods private
- Update CHANGELOG.md

---

## Testing Strategy

After each phase:
1. **Compile:** `sbt compile` - ensure all projects compile
2. **Test:** `sbt test` - ensure all 818 tests pass
3. **Commit:** Small, focused commits per phase

---

## Success Criteria

✅ **menger-common module created** with Vector, Const, ImageSize, Light, Vec3
✅ **No circular dependencies** - clean module structure
✅ **Unified Light types** - single implementation, no duplication
✅ **Type-safe coordinates** - Vector[3] with .x, .y, .z accessors
✅ **Incremental migration** - deprecation warnings guide test updates
✅ **All tests pass** - 818 tests pass after each phase

---

## Estimated Effort

- **Phase 1-3:** 2-3 hours (module setup, core types)
- **Phase 4-5:** 1-2 hours (Vector[3] infrastructure)
- **Phase 6:** 3-5 hours (test migration, can be incremental)
- **Phase 7:** 1 hour (cleanup)

**Total:** 7-11 hours over multiple commits
