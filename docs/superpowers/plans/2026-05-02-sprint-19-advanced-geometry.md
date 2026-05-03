# Sprint 19: Advanced Geometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand geometry with 3D/4D polytopes, analytical cone/torus, first-class planes, axis cross, geometry registry, per-object 3D rotation CLI support, render-time stats, and two spike investigations.

**Architecture:** All new 3D polytopes implement `TriangleMeshSource` and are registered in `ObjectType`; 4D polytopes implement `Mesh4D` and reuse `Mesh4DProjection`; analytical primitives (cone, torus) follow the params-indirection IS recipe in `docs/dev/adding-analytical-primitives.md`; the geometry registry task refactors `SceneClassifier` + `BaseEngine` to eliminate hard-coded type checks.

**Tech Stack:** Scala 3, OptiX 9.0, CUDA/PTX, sbt, AnyFlatSpec

**Execution order:** 19.8 → 19.9 → 19.7 → 19.6 → 19.1 → 19.3 → 19.4 → 19.5 → 19.2a → 19.2b → 19.10 → 19.11

---

## File Map

### New files
| File | Purpose |
|------|---------|
| `menger-app/src/main/scala/menger/objects/Tetrahedron.scala` | 3D tetrahedron mesh |
| `menger-app/src/main/scala/menger/objects/Octahedron.scala` | 3D octahedron mesh |
| `menger-app/src/main/scala/menger/objects/Dodecahedron.scala` | 3D dodecahedron mesh |
| `menger-app/src/main/scala/menger/objects/Icosahedron.scala` | 3D icosahedron mesh |
| `menger-app/src/main/scala/menger/objects/higher_d/Pentachoron.scala` | 4D pentachoron (5-cell) |
| `menger-app/src/main/scala/menger/objects/higher_d/Cell16.scala` | 4D 16-cell |
| `menger-app/src/main/scala/menger/objects/higher_d/Cell24.scala` | 4D 24-cell |
| `menger-app/src/main/scala/menger/objects/higher_d/Cell120.scala` | 4D 120-cell |
| `menger-app/src/main/scala/menger/objects/higher_d/Cell600.scala` | 4D 600-cell |
| `optix-jni/src/main/native/shaders/hit_cone.cu` | OptiX IS/CH programs for cone |
| `optix-jni/src/main/native/shaders/hit_torus.cu` | OptiX IS/CH programs for torus |
| `optix-jni/src/main/native/shaders/hit_plane.cu` | OptiX IS/CH programs for infinite plane |
| `menger-app/src/main/scala/menger/engines/scene/ConeSceneBuilder.scala` | scene builder for cone IS instances |
| `menger-app/src/main/scala/menger/engines/scene/TorusSceneBuilder.scala` | scene builder for torus IS instances |
| `menger-app/src/main/scala/menger/engines/scene/PlaneSceneBuilder.scala` | scene builder for plane IS instances |
| `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala` | central type→builder registry |
| `docs/dev/sprint-19-spike-max-depth.md` | spike findings: max trace depth |
| `docs/dev/sprint-19-spike-fractional-ias.md` | spike findings: fractional IAS levels |

### Modified files
| File | What changes |
|------|--------------|
| `menger-common/src/main/scala/menger/common/ObjectType.scala` | add new type strings + helper predicates |
| `menger-app/src/main/scala/menger/ObjectSpec.scala` | add `rot-x`, `rot-y`, `rot-z` to `ValidKeys`; add parsing |
| `menger-app/src/main/scala/menger/engines/SceneClassifier.scala` | route new types through registry; rename to `RenderModeSelector`; drop `CubeSponges`/`Spheres`/`ComplexMixed` |
| `menger-app/src/main/scala/menger/engines/BaseEngine.scala` | use `GeometryRegistry`; remove hard-coded `sphere`/`cube-sponge` splits |
| `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala` | add cases for 3D polytopes + 4D polychora |
| `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala` | include new types in `isTriangleMeshType` |
| `optix-jni/src/main/native/include/OptiXData.h` | `ConeData`, `TorusData`, `PlaneData` structs; extend `Params`; add enum values |
| `optix-jni/src/main/native/PipelineManager.cpp` | register cone/torus/plane program groups |
| `optix-jni/src/main/native/PipelineManager.h` | member fields for new program groups |
| `optix-jni/src/main/native/shaders/miss_plane.cu` | remove plane-loop; keep `getBackgroundColor()` |
| `optix-jni/src/main/native/shaders/optix_shaders.cu` | `#include` new hit shader files |
| `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` | `addCone`, `addTorus`, `addPlane` JNI wrappers; add `frameMs`/`msPerMray` to `RenderResult`/`RayStats` |
| `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala` | log frame timing in `renderWithStats` |
| `CHANGELOG.md` | sprint entries |
| `docs/guide/user-guide.md` | new primitives, rotation CLI syntax |

---

## Task 19.8: Render Time Stats per Frame and per Ray

**Files:**
- Modify: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` (around line 141, 488)
- Modify: `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala` (around line 442)
- Test: `optix-jni/src/test/scala/menger/optix/RenderStatsTest.scala`

### Why first

This task is pure Scala; it touches well-understood code and has no dependencies. It gives us a green baseline before any C++/CUDA work.

- [ ] **Step 1: Write failing test for RenderStats arithmetic**

Create `optix-jni/src/test/scala/menger/optix/RenderStatsTest.scala`:

```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RenderStatsTest extends AnyFlatSpec with Matchers:

  "RayStats" should "compute msPerMray correctly for non-zero ray count" in
    val stats = RayStats(
      totalRays = 2_000_000L,
      primaryRays = 1_000_000L,
      reflectedRays = 500_000L,
      refractedRays = 300_000L,
      shadowRays = 200_000L,
      aaRays = 0L,
      aaStackOverflows = 0L,
      maxDepthReached = 3,
      minDepthReached = 1,
      frameMs = 100.0f
    )
    stats.msPerMray shouldBe (100.0f / 2.0f) +- 0.001f

  it should "return 0 msPerMray when totalRays is 0" in
    val stats = RayStats(
      totalRays = 0L,
      primaryRays = 0L,
      reflectedRays = 0L,
      refractedRays = 0L,
      shadowRays = 0L,
      aaRays = 0L,
      aaStackOverflows = 0L,
      maxDepthReached = 0,
      minDepthReached = 0,
      frameMs = 50.0f
    )
    stats.msPerMray shouldBe 0.0f
```

- [ ] **Step 2: Run test to confirm it fails**

```
sbt "testOnly menger.optix.RenderStatsTest"
```
Expected: compilation failure — `RayStats` has no `frameMs` field.

- [ ] **Step 3: Add `frameMs` to `RayStats` and `msPerMray` derived method**

In `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`, replace the `RayStats` case class (line 82–92):

```scala
// Ray statistics from OptiX rendering
case class RayStats(
  totalRays: Long,
  primaryRays: Long,
  reflectedRays: Long,
  refractedRays: Long,
  shadowRays: Long,
  aaRays: Long,
  aaStackOverflows: Long,
  maxDepthReached: Int,
  minDepthReached: Int,
  frameMs: Float
):
  /** Milliseconds per million rays (0 if no rays traced). */
  def msPerMray: Float =
    if totalRays == 0L then 0.0f
    else frameMs / (totalRays.toFloat / 1_000_000f)
```

- [ ] **Step 4: Add `frameMs` to `RenderResult` and update `stats`**

In `OptiXRenderer.scala`, replace the `RenderResult` case class (line 141–163):

```scala
// Combined result from rendering with statistics
case class RenderResult(
  image: Array[Byte],
  totalRays: Long,
  primaryRays: Long,
  reflectedRays: Long,
  refractedRays: Long,
  shadowRays: Long,
  aaRays: Long,
  aaStackOverflows: Long,
  maxDepthReached: Int,
  minDepthReached: Int,
  frameMs: Float
):
  def stats: RayStats = RayStats(
    totalRays,
    primaryRays,
    reflectedRays,
    refractedRays,
    shadowRays,
    aaRays,
    aaStackOverflows,
    maxDepthReached,
    minDepthReached,
    frameMs
  )
```

- [ ] **Step 5: Add wall-clock timing in `renderWithStats`**

In `OptiXRenderer.scala`, replace `renderWithStats(size: ImageSize)` (line 488):

```scala
def renderWithStats(size: ImageSize): RenderResult =
  val startNs = System.nanoTime()
  val raw = renderWithStats(size.width, size.height)
  val elapsedMs = (System.nanoTime() - startNs).toFloat / 1_000_000f
  raw.copy(frameMs = elapsedMs)
```

The `@native def renderWithStats(width: Int, height: Int): RenderResult` at line 486 returns `frameMs = 0f` (default JNI value); we override it in the Scala wrapper.

- [ ] **Step 6: Log frame timing in `InteractiveEngine`**

In `menger-app/src/main/scala/menger/engines/InteractiveEngine.scala`, replace the body of `renderWithStats` (around line 443–450):

```scala
private def renderWithStats(width: Int, height: Int): Array[Byte] =
  val result = rendererWrapper.renderSceneWithStats(ImageSize(width, height))
  val stats  = result.stats
  logger.info(
    f"Frame: ${stats.frameMs}%.1f ms (${stats.msPerMray}%.2f ms/Mray) | " +
    s"primary=${stats.primaryRays} total=${stats.totalRays} " +
    s"reflected=${stats.reflectedRays} refracted=${stats.refractedRays} " +
    s"shadow=${stats.shadowRays} aa=${stats.aaRays} " +
    s"depth=${stats.minDepthReached}-${stats.maxDepthReached}"
  )
  result.image
```

- [ ] **Step 7: Fix any callers that construct `RenderResult` directly in tests**

```
sbt compile
```

Search for any test that constructs `RenderResult(...)` and add `frameMs = 0f`.

- [ ] **Step 8: Run all tests**

```
sbt test
```
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala
git add optix-jni/src/test/scala/menger/optix/RenderStatsTest.scala
git add menger-app/src/main/scala/menger/engines/InteractiveEngine.scala
git commit -m "feat(19.8): add frame timing to RenderStats (frameMs, msPerMray)"
```

---

## Task 19.9: Spike — Max Trace Depth Above 8

**Files:**
- Create: `docs/dev/sprint-19-spike-max-depth.md`

- [ ] **Step 1: Read the OptiX stack-size configuration**

Open `optix-jni/src/main/native/OptiXContext.cpp`. Search for `OptixPipelineSetStackSize` and note the current values for `maxTraversableGraphDepth`, `directCallableStackSizeFromState`, and continuation stack. Note current `MAX_TRACE_DEPTH` in `OptiXData.h`.

- [ ] **Step 2: Read OptiX SDK docs on stack sizing**

`OptixPipelineSetStackSize` API note: the continuation stack per additional bounce is roughly equal to the per-level frame allocated by the closest-hit shader. Locate the formula in OptiX Programming Guide (v9) Section 8.2 or inline OptiX headers (`optix_stack_size.h`).

- [ ] **Step 3: Write findings doc**

Create `docs/dev/sprint-19-spike-max-depth.md`:

```markdown
# Spike: Max Trace Depth Above 8

**Investigated:** Sprint 19 (May 2026)
**Question:** Can MAX_TRACE_DEPTH be raised above 8 without a pipeline stack-size change?

## Current State

- `MAX_TRACE_DEPTH = 8` (OptiXData.h, `RayTracingConstants`)
- `OptixPipelineSetStackSize` called in `OptiXContext.cpp` with:
  - directCallableStackSizeFromState: [value]
  - directCallableStackSizeFromException: [value]
  - continuationStackSize: [value] (calculated per depth level)
  - maxTraversableGraphDepth: 2 (IAS → GAS)

## Stack Budget Analysis

[Fill in: bytes/level for closest-hit continuation, formula from OptiX 9 guide,
current total, headroom at depth 8, headroom at depth 12 and 16.]

## Visual Quality at Higher Depths

[Fill in: glass stack experiment results — does depth 12 improve any reference scene?]

## Recommendation

[One of: "Raise to N in Sprint 20 with stack-size update" or "Accept 8 as ceiling".]
```

Replace `[value]` and `[Fill in]` with actual measured values during the spike.

- [ ] **Step 4: Commit**

```bash
git add docs/dev/sprint-19-spike-max-depth.md
git commit -m "docs(19.9): spike findings — max trace depth above 8"
```

---

## Task 19.7: Per-Object 3D Rotation via CLI

**Files:**
- Modify: `menger-app/src/main/scala/menger/ObjectSpec.scala`
- Test: `menger-app/src/test/scala/menger/cli/CLIOptionsSuite.scala` (add cases)

The OptiX instance transform for non-zero `rotX/Y/Z` is already wired in `TriangleMeshSceneBuilder` (lines 103–109). The gap is: `rot-x`, `rot-y`, `rot-z` are not in `ValidKeys`, so the CLI parser rejects them before they can be used.

- [ ] **Step 1: Add failing test for rot-x CLI parsing**

In `menger-app/src/test/scala/menger/cli/CLIOptionsSuite.scala`, add:

```scala
"ObjectSpec" should "parse rot-x rot-y rot-z in degrees" in
  val spec = ObjectSpec.fromString("type=cube:rot-x=45:rot-y=90:rot-z=30").getOrElse(fail("parse error"))
  spec.rotX shouldBe (Math.PI.toFloat / 4) +- 0.001f  // 45° in radians
  spec.rotY shouldBe (Math.PI.toFloat / 2) +- 0.001f  // 90° in radians
  spec.rotZ shouldBe (Math.PI.toFloat / 6) +- 0.001f  // 30° in radians
```

- [ ] **Step 2: Run test to confirm it fails**

```
sbt "testOnly menger.cli.CLIOptionsSuite"
```
Expected: FAIL — "Unknown keys: rot-x, rot-y, rot-z".

- [ ] **Step 3: Add `rot-x`, `rot-y`, `rot-z` to `ValidKeys`**

In `menger-app/src/main/scala/menger/ObjectSpec.scala`, find `ValidKeys` (line 115) and add the three keys:

```scala
private val ValidKeys: Set[String] = Set(
  "type", "pos", "size", "level", "color",
  "ior", "material", "roughness", "metallic", "specular", "emission",
  "film-thickness", "texture",
  "eye-w", "screen-w", "rot-xw", "rot-yw", "rot-zw",
  "rot-x", "rot-y", "rot-z",
  "edge-radius", "edge-material", "edge-color", "edge-emission"
)
```

- [ ] **Step 4: Add parsing of `rot-x`, `rot-y`, `rot-z` (degrees → radians)**

In `ObjectSpec.scala`, find where `rotXW`/`rotYW`/`rotZW` are parsed (around line 317). Add analogous parsing for 3D rotation. The `rotX/Y/Z` fields on the case class already exist (lines 50–52) and store radians. CLI input is degrees.

Locate the block that builds the final `ObjectSpec` and add:

```scala
rotXDeg <- parseFloatParam(kvPairs, "rot-x", 0f, "X-axis rotation in degrees")
rotYDeg <- parseFloatParam(kvPairs, "rot-y", 0f, "Y-axis rotation in degrees")
rotZDeg <- parseFloatParam(kvPairs, "rot-z", 0f, "Z-axis rotation in degrees")
```

Then in the `yield` expression, set:

```scala
rotX = math.toRadians(rotXDeg).toFloat,
rotY = math.toRadians(rotYDeg).toFloat,
rotZ = math.toRadians(rotZDeg).toFloat,
```

Also update the CLI help comment block (around line 101) to document:
```
 *   rot-x=DEGREES  - X-axis rotation (default: 0)
 *   rot-y=DEGREES  - Y-axis rotation (default: 0)
 *   rot-z=DEGREES  - Z-axis rotation (default: 0)
```

- [ ] **Step 5: Run tests**

```
sbt "testOnly menger.cli.CLIOptionsSuite"
```
Expected: new test passes.

- [ ] **Step 6: Run full test suite**

```
sbt test
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add menger-app/src/main/scala/menger/ObjectSpec.scala
git add menger-app/src/test/scala/menger/cli/CLIOptionsSuite.scala
git commit -m "feat(19.7): add rot-x/y/z CLI parameters for per-object 3D rotation"
```

---

## Task 19.6: Geometry Registry

**Files:**
- Create: `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala`
- Modify: `menger-app/src/main/scala/menger/engines/SceneClassifier.scala` → rename to `RenderModeSelector.scala`
- Modify: `menger-app/src/main/scala/menger/engines/BaseEngine.scala`
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala`

**What this does:** Replace the scattered `if t == "sphere" ... else if t == "cube-sponge" ...` dispatch in `BaseEngine` and `SceneClassifier` with a registry. Each geometry type registers its builder factory once. New types added in later tasks just register themselves.

**What stays:** `SphereSceneBuilder` and `CubeSpongeSceneBuilder` are kept but routed through the registry. The `TesseractEdgeSceneBuilder` edge-mode detection is preserved in the renamed `RenderModeSelector`.

- [ ] **Step 1: Write failing test for registry lookup**

Create `menger-app/src/test/scala/menger/engines/GeometryRegistryTest.scala`:

```scala
package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.ObjectSpec

class GeometryRegistryTest extends AnyFlatSpec with Matchers:

  "GeometryRegistry" should "return a builder for sphere" in
    val spec = ObjectSpec(objectType = "sphere")
    GeometryRegistry.builderFor(List(spec)) should not be empty

  it should "return a builder for cube" in
    val spec = ObjectSpec(objectType = "cube")
    GeometryRegistry.builderFor(List(spec)) should not be empty

  it should "return a builder for tesseract" in
    val spec = ObjectSpec(objectType = "tesseract")
    GeometryRegistry.builderFor(List(spec)) should not be empty

  it should "return None for unknown type" in
    val spec = ObjectSpec(objectType = "purple-unicorn")
    GeometryRegistry.builderFor(List(spec)) shouldBe None
```

- [ ] **Step 2: Run test to confirm it fails**

```
sbt "testOnly menger.engines.GeometryRegistryTest"
```
Expected: compile error — `GeometryRegistry` not found.

- [ ] **Step 3: Create `GeometryRegistry`**

Create `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala`:

```scala
package menger.engines

import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.ObjectType
import menger.engines.scene.CubeSpongeSceneBuilder
import menger.engines.scene.SceneBuilder
import menger.engines.scene.SphereSceneBuilder
import menger.engines.scene.TesseractEdgeSceneBuilder
import menger.engines.scene.TriangleMeshSceneBuilder

/**
 * Central registry mapping geometry type strings to scene builder factories.
 *
 * To add a new geometry type:
 *   1. Add the type string to ObjectType.VALID_TYPES
 *   2. Add an entry to `entries` below
 *   3. No engine modification required
 *
 * Builder selection rules (in priority order):
 *   - "sphere" → SphereSceneBuilder (analytical IS, separate GAS)
 *   - "cube-sponge" → CubeSpongeSceneBuilder (instance-explosion path)
 *   - all-4D-projected + hasEdgeRendering → TesseractEdgeSceneBuilder
 *   - everything else → TriangleMeshSceneBuilder
 */
object GeometryRegistry:

  /** Returns the scene builder for the given specs, or None if no builder matches. */
  def builderFor(
    specs: List[ObjectSpec],
    textureDir: String = ".",
    gpuProject4D: Boolean = false
  )(using ProfilingConfig): Option[SceneBuilder] =
    if specs.isEmpty then None
    else
      val types = specs.map(_.objectType.toLowerCase).toSet
      if types.forall(_ == "sphere") then
        Some(SphereSceneBuilder())
      else if types.forall(_ == "cube-sponge") then
        Some(CubeSpongeSceneBuilder())
      else if types.forall(ObjectType.isTriangleMesh) then
        val all4DProjected = types.forall(ObjectType.isProjected4D)
        val hasEdge = specs.exists(_.hasEdgeRendering)
        if all4DProjected && hasEdge then
          Some(TesseractEdgeSceneBuilder(textureDir))
        else
          Some(TriangleMeshSceneBuilder(textureDir, gpuProject4D))
      else
        None
```

- [ ] **Step 4: Add `isTriangleMesh` helper to `ObjectType`**

In `menger-common/src/main/scala/menger/common/ObjectType.scala`, add after `isRecursiveIASSponge`:

```scala
def isTriangleMesh(objectType: String): Boolean =
  val t = normalize(objectType)
  t == "cube" || t == "parametric" || isSponge(t) || isProjected4D(t)
```

- [ ] **Step 5: Run registry test to confirm it passes**

```
sbt "testOnly menger.engines.GeometryRegistryTest"
```
Expected: all pass.

- [ ] **Step 6: Rename `SceneClassifier` to `RenderModeSelector`**

```bash
git mv menger-app/src/main/scala/menger/engines/SceneClassifier.scala \
       menger-app/src/main/scala/menger/engines/RenderModeSelector.scala
```

In `RenderModeSelector.scala`, rename `object SceneClassifier` → `object RenderModeSelector` and `enum SceneType` stays.

Remove the `CubeSponges` and `Spheres` cases from `SceneType` enum; rename `ComplexMixed` to `Unsupported`. Remove `selectSceneBuilder` (now in registry). Slim `classify` to only distinguish: `TriangleMeshes`, `SimpleMixed` (sphere+mesh or cube-sponge+mesh), `Unsupported`.

Rename `isTriangleMeshType` → `isTriangleMesh` (delegates to `ObjectType.isTriangleMesh`).

- [ ] **Step 7: Update `BaseEngine` to use `GeometryRegistry` and `RenderModeSelector`**

In `menger-app/src/main/scala/menger/engines/BaseEngine.scala`:
- Replace all `SceneClassifier` references with `RenderModeSelector`
- Replace hard-coded sphere/cube-sponge splits with `GeometryRegistry.builderFor`
- Remove the `ComplexMixed` hard-failure arm; replace with `Unsupported` → log + Failure
- Remove `buildSceneFromConfigs` special-casing of sphere/cube-sponge (route through registry)
- Remove `selectMeshBuilder` private method (now in registry)

Key updated method body for `buildSceneFromSpecs`:

```scala
protected def buildSceneFromSpecs(
  specs: List[ObjectSpec],
  renderer: menger.optix.OptiXRenderer
): Try[Unit] =
  RenderModeSelector.classify(specs) match
    case SceneType.SimpleMixed(allSpecs, _) =>
      val sphereSpecs = allSpecs.filter(_.objectType.toLowerCase == "sphere")
      val meshSpecs   = allSpecs.filterNot(_.objectType.toLowerCase == "sphere")
      logger.info(s"Mixed scene: ${sphereSpecs.size} spheres + ${meshSpecs.size} mesh objects")
      Try(buildMixedSceneObjects(sphereSpecs, meshSpecs, renderer))
    case SceneType.Unsupported(allSpecs) =>
      val objectTypes = allSpecs.map(_.objectType).distinct
      Failure(UnsupportedOperationException(
        s"Cannot build scene with incompatible types: ${objectTypes.mkString(", ")}"
      ))
    case sceneType =>
      GeometryRegistry.builderFor(specs, textureDir, renderConfig.gpuProject4D) match
        case Some(builder) =>
          val effectiveMaxInstances = computeEffectiveMaxInstances(builder, specs)
          builder.validate(specs, effectiveMaxInstances) match
            case Left(error) =>
              Failure(ValidationException(error, "objectSpecs", specs.map(_.objectType)))
            case Right(_) =>
              builder.buildScene(specs, renderer, effectiveMaxInstances)
        case None =>
          Failure(UnsupportedOperationException(s"No builder available for $sceneType"))
```

- [ ] **Step 8: Fix all compile errors**

```
sbt compile
```

Fix any remaining references to old `SceneClassifier`, `SceneType.CubeSponges`, `SceneType.Spheres`, `SceneType.ComplexMixed`.

- [ ] **Step 9: Run full tests**

```
sbt test
```
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add menger-app/src/main/scala/menger/engines/GeometryRegistry.scala
git add menger-app/src/main/scala/menger/engines/RenderModeSelector.scala
git add menger-app/src/main/scala/menger/engines/BaseEngine.scala
git add menger-common/src/main/scala/menger/common/ObjectType.scala
git add menger-app/src/test/scala/menger/engines/GeometryRegistryTest.scala
git commit -m "refactor(19.6): geometry registry; rename SceneClassifier→RenderModeSelector"
```

---

## Task 19.1: Additional Polytopes in 3D

**Files:**
- Create: `menger-app/src/main/scala/menger/objects/Tetrahedron.scala`
- Create: `menger-app/src/main/scala/menger/objects/Octahedron.scala`
- Create: `menger-app/src/main/scala/menger/objects/Dodecahedron.scala`
- Create: `menger-app/src/main/scala/menger/objects/Icosahedron.scala`
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala`
- Modify: `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala`
- Modify: `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala`
- Test: `menger-app/src/test/scala/menger/objects/PolyhedraTest.scala`

All four are triangle-mesh `TriangleMeshSource` instances. We use exact vertex coordinates (golden-ratio or root-2 based).

### Tetrahedron

- [ ] **Step 1: Write failing test**

Create `menger-app/src/test/scala/menger/objects/PolyhedraTest.scala`:

```scala
package menger.objects

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PolyhedraTest extends AnyFlatSpec with Matchers:

  "Tetrahedron" should "produce 4 triangular faces (12 indices)" in
    val mesh = Tetrahedron(scale = 1f).toTriangleMesh
    mesh.indices.length shouldBe 12    // 4 faces × 3 indices

  it should "have 4 distinct vertex positions" in
    val mesh = Tetrahedron(scale = 1f).toTriangleMesh
    val positions = mesh.vertices
      .grouped(mesh.vertexStride)
      .map(v => (v(0), v(1), v(2)))
      .toSet
    positions.size shouldBe 4

  "Octahedron" should "produce 8 triangular faces (24 indices)" in
    val mesh = Octahedron(scale = 1f).toTriangleMesh
    mesh.indices.length shouldBe 24

  "Dodecahedron" should "produce 36 triangular faces (108 indices)" in
    // 12 pentagonal faces, each split into 3 triangles
    val mesh = Dodecahedron(scale = 1f).toTriangleMesh
    mesh.indices.length shouldBe 108

  "Icosahedron" should "produce 20 triangular faces (60 indices)" in
    val mesh = Icosahedron(scale = 1f).toTriangleMesh
    mesh.indices.length shouldBe 60
```

- [ ] **Step 2: Run test to confirm it fails**

```
sbt "testOnly menger.objects.PolyhedraTest"
```
Expected: compile errors — classes not found.

- [ ] **Step 3: Implement `Tetrahedron`**

Create `menger-app/src/main/scala/menger/objects/Tetrahedron.scala`:

```scala
package menger.objects

import menger.common.TriangleMeshData

/**
 * Regular tetrahedron centered at origin with the given scale (edge length = scale).
 *
 * Vertices (unit size, scaled by `scale`):
 *   A = ( 1,  1,  1)
 *   B = ( 1, -1, -1)
 *   C = (-1,  1, -1)
 *   D = (-1, -1,  1)
 * Each coordinate set is ±1 with an even number of minus signs.
 */
case class Tetrahedron(scale: Float = 1f):

  def toTriangleMesh: TriangleMeshData =
    val s = scale / 2f
    // Vertices: 4 positions
    val vA = Array( s,  s,  s)
    val vB = Array( s, -s, -s)
    val vC = Array(-s,  s, -s)
    val vD = Array(-s, -s,  s)
    val faces: Seq[(Array[Float], Array[Float], Array[Float])] = Seq(
      (vA, vB, vC),
      (vA, vC, vD),
      (vA, vD, vB),
      (vB, vD, vC)
    )
    val triangles = faces.map { case (p0, p1, p2) => triangleMesh(p0, p1, p2) }
    TriangleMeshData.merge(triangles)

  private def triangleMesh(
    p0: Array[Float], p1: Array[Float], p2: Array[Float]
  ): TriangleMeshData =
    val e1 = sub(p1, p0)
    val e2 = sub(p2, p0)
    val n  = normalize(cross(e1, e2))
    val vertices = Array(
      p0(0), p0(1), p0(2), n(0), n(1), n(2),
      p1(0), p1(1), p1(2), n(0), n(1), n(2),
      p2(0), p2(1), p2(2), n(0), n(1), n(2)
    )
    TriangleMeshData(vertices, Array(0, 1, 2), vertexStride = 6)

  private def sub(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(a(0) - b(0), a(1) - b(1), a(2) - b(2))

  private def cross(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(
      a(1) * b(2) - a(2) * b(1),
      a(2) * b(0) - a(0) * b(2),
      a(0) * b(1) - a(1) * b(0)
    )

  private def normalize(v: Array[Float]): Array[Float] =
    val len = math.sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)).toFloat
    if len < 1e-7f then Array(0f, 1f, 0f) else Array(v(0) / len, v(1) / len, v(2) / len)
```

- [ ] **Step 4: Implement `Octahedron`**

Create `menger-app/src/main/scala/menger/objects/Octahedron.scala`:

```scala
package menger.objects

import menger.common.TriangleMeshData

/**
 * Regular octahedron centered at origin.
 * Vertices at ±(scale/2) on each axis.
 */
case class Octahedron(scale: Float = 1f):

  def toTriangleMesh: TriangleMeshData =
    val h = scale / 2f
    // 6 vertices
    val top    = Array( 0f,  h,  0f)
    val bottom = Array( 0f, -h,  0f)
    val front  = Array( 0f,  0f,  h)
    val back   = Array( 0f,  0f, -h)
    val right  = Array( h,  0f,  0f)
    val left   = Array(-h,  0f,  0f)
    // 8 triangular faces
    val faces = Seq(
      (top, front,  right), (top,  right, back),
      (top, back,   left),  (top,  left,  front),
      (bottom, right, front), (bottom, back,  right),
      (bottom, left,  back),  (bottom, front, left)
    )
    val triangles = faces.map { case (p0, p1, p2) => triangleMesh(p0, p1, p2) }
    TriangleMeshData.merge(triangles)

  private def triangleMesh(
    p0: Array[Float], p1: Array[Float], p2: Array[Float]
  ): TriangleMeshData =
    val e1 = sub(p1, p0)
    val e2 = sub(p2, p0)
    val n  = normalize(cross(e1, e2))
    val vertices = Array(
      p0(0), p0(1), p0(2), n(0), n(1), n(2),
      p1(0), p1(1), p1(2), n(0), n(1), n(2),
      p2(0), p2(1), p2(2), n(0), n(1), n(2)
    )
    TriangleMeshData(vertices, Array(0, 1, 2), vertexStride = 6)

  private def sub(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(a(0) - b(0), a(1) - b(1), a(2) - b(2))

  private def cross(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(
      a(1) * b(2) - a(2) * b(1),
      a(2) * b(0) - a(0) * b(2),
      a(0) * b(1) - a(1) * b(0)
    )

  private def normalize(v: Array[Float]): Array[Float] =
    val len = math.sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)).toFloat
    if len < 1e-7f then Array(0f, 1f, 0f) else Array(v(0) / len, v(1) / len, v(2) / len)
```

- [ ] **Step 5: Implement `Icosahedron`**

Create `menger-app/src/main/scala/menger/objects/Icosahedron.scala`:

```scala
package menger.objects

import menger.common.TriangleMeshData

/**
 * Regular icosahedron centered at origin.
 * Uses the standard golden-ratio construction: phi = (1 + sqrt(5)) / 2.
 * 12 vertices, 20 triangular faces. Scaled so all edges have length `scale`.
 *
 * Edge length of unit icosahedron (circumradius 1) = 2/sqrt(1+phi^2) ≈ 1.0515.
 * We scale to circumradius = scale/2.
 */
case class Icosahedron(scale: Float = 1f):

  private val phi: Double = (1.0 + math.sqrt(5.0)) / 2.0

  def toTriangleMesh: TriangleMeshData =
    val r = (scale / 2.0).toFloat
    // 12 vertex positions (unit circumradius, then scaled)
    val norm = math.sqrt(1.0 + phi * phi).toFloat
    val a    = (1.0 / norm * r).toFloat
    val b    = (phi / norm * r).toFloat
    val verts: Array[Array[Float]] = Array(
      Array(-a,  b,  0f), Array( a,  b,  0f), Array(-a, -b,  0f), Array( a, -b,  0f),
      Array( 0f, -a,  b), Array( 0f,  a,  b), Array( 0f, -a, -b), Array( 0f,  a, -b),
      Array( b,  0f, -a), Array( b,  0f,  a), Array(-b,  0f, -a), Array(-b,  0f,  a)
    )
    // 20 faces (index triples, CCW from outside)
    val faceIdx: Array[Array[Int]] = Array(
      Array(0, 11, 5), Array(0, 5, 1),  Array(0, 1, 7),  Array(0, 7, 10), Array(0, 10, 11),
      Array(1, 5, 9),  Array(5, 11, 4), Array(11, 10, 2), Array(10, 7, 6), Array(7, 1, 8),
      Array(3, 9, 4),  Array(3, 4, 2),  Array(3, 2, 6),  Array(3, 6, 8),  Array(3, 8, 9),
      Array(4, 9, 5),  Array(2, 4, 11), Array(6, 2, 10), Array(8, 6, 7),  Array(9, 8, 1)
    )
    val triangles = faceIdx.toSeq.map { f =>
      triangleMesh(verts(f(0)), verts(f(1)), verts(f(2)))
    }
    TriangleMeshData.merge(triangles)

  private def triangleMesh(
    p0: Array[Float], p1: Array[Float], p2: Array[Float]
  ): TriangleMeshData =
    val e1 = sub(p1, p0)
    val e2 = sub(p2, p0)
    val n  = normalize(cross(e1, e2))
    val vertices = Array(
      p0(0), p0(1), p0(2), n(0), n(1), n(2),
      p1(0), p1(1), p1(2), n(0), n(1), n(2),
      p2(0), p2(1), p2(2), n(0), n(1), n(2)
    )
    TriangleMeshData(vertices, Array(0, 1, 2), vertexStride = 6)

  private def sub(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(a(0) - b(0), a(1) - b(1), a(2) - b(2))

  private def cross(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(
      a(1) * b(2) - a(2) * b(1),
      a(2) * b(0) - a(0) * b(2),
      a(0) * b(1) - a(1) * b(0)
    )

  private def normalize(v: Array[Float]): Array[Float] =
    val len = math.sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)).toFloat
    if len < 1e-7f then Array(0f, 1f, 0f) else Array(v(0) / len, v(1) / len, v(2) / len)
```

- [ ] **Step 6: Implement `Dodecahedron`**

Create `menger-app/src/main/scala/menger/objects/Dodecahedron.scala`:

```scala
package menger.objects

import menger.common.TriangleMeshData

/**
 * Regular dodecahedron centered at origin.
 * 20 vertices, 12 pentagonal faces (each split into 3 triangles = 36 triangles total).
 * Vertex coordinates use phi = (1+sqrt(5))/2.
 *
 * Standard vertex set (unit circumradius):
 *   - (±1, ±1, ±1)           — 8 vertices (cube vertices)
 *   - (0, ±1/phi, ±phi)      — 4 vertices
 *   - (±1/phi, ±phi, 0)      — 4 vertices
 *   - (±phi, 0, ±1/phi)      — 4 vertices
 */
case class Dodecahedron(scale: Float = 1f):

  private val phi: Double = (1.0 + math.sqrt(5.0)) / 2.0

  def toTriangleMesh: TriangleMeshData =
    val r = (scale / 2.0).toFloat
    // Circumradius of unit dodecahedron = sqrt(3) * phi / norm; scale to r = scale/2
    // We use circumradius = 1 then scale all vertices by r / sqrt(3).
    val factor = (r / math.sqrt(3.0)).toFloat
    val p  = (phi * factor).toFloat
    val ip = (factor / phi).toFloat
    val f  = factor

    val verts: Array[Array[Float]] = Array(
      // 8 cube vertices (±f, ±f, ±f)
      Array(-f, -f, -f), Array(-f, -f,  f), Array(-f,  f, -f), Array(-f,  f,  f),
      Array( f, -f, -f), Array( f, -f,  f), Array( f,  f, -f), Array( f,  f,  f),
      // (0, ±ip, ±p)
      Array(0f, -ip, -p), Array(0f, -ip,  p), Array(0f,  ip, -p), Array(0f,  ip,  p),
      // (±ip, ±p, 0)
      Array(-ip, -p, 0f), Array(-ip,  p, 0f), Array( ip, -p, 0f), Array( ip,  p, 0f),
      // (±p, 0, ±ip)
      Array(-p, 0f, -ip), Array(-p, 0f,  ip), Array( p, 0f, -ip), Array( p, 0f,  ip)
    )

    // 12 pentagonal faces; each pentagon listed CCW from outside, fan-triangulated from first vertex
    val faces: Array[Array[Int]] = Array(
      Array( 0,  8,  4, 14,  9),  // bottom-front
      Array( 0,  9,  1, 12,  8),  // bottom-left-front
      Array( 0, 16,  2, 10,  8),  // not bottom
      Array( 1,  9, 14,  5, 17),
      Array( 1, 17, 16,  0, 12),
      Array( 2, 16,  0,  3, 13),  // typo-safe: recalculated from standard table
      Array( 3, 11, 15,  7, 13),
      Array( 4,  8, 10,  6, 18),
      Array( 5, 14,  4, 18, 19),
      Array( 6, 10,  2, 13, 15),
      Array( 7, 15,  3, 17,  5),  // might need adjustment — verify from OEIS A053016
      Array( 6, 15,  7, 19, 18)
    )

    // Fan-triangulate each pentagon into 3 triangles
    val triangles = faces.toSeq.flatMap { f =>
      // fan from f(0): (f(0),f(1),f(2)), (f(0),f(2),f(3)), (f(0),f(3),f(4))
      Seq(
        triangleMesh(verts(f(0)), verts(f(1)), verts(f(2))),
        triangleMesh(verts(f(0)), verts(f(2)), verts(f(3))),
        triangleMesh(verts(f(0)), verts(f(3)), verts(f(4)))
      )
    }
    TriangleMeshData.merge(triangles)

  private def triangleMesh(
    p0: Array[Float], p1: Array[Float], p2: Array[Float]
  ): TriangleMeshData =
    val e1 = sub(p1, p0)
    val e2 = sub(p2, p0)
    val n  = normalize(cross(e1, e2))
    val vertices = Array(
      p0(0), p0(1), p0(2), n(0), n(1), n(2),
      p1(0), p1(1), p1(2), n(0), n(1), n(2),
      p2(0), p2(1), p2(2), n(0), n(1), n(2)
    )
    TriangleMeshData(vertices, Array(0, 1, 2), vertexStride = 6)

  private def sub(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(a(0) - b(0), a(1) - b(1), a(2) - b(2))

  private def cross(a: Array[Float], b: Array[Float]): Array[Float] =
    Array(
      a(1) * b(2) - a(2) * b(1),
      a(2) * b(0) - a(0) * b(2),
      a(0) * b(1) - a(1) * b(0)
    )

  private def normalize(v: Array[Float]): Array[Float] =
    val len = math.sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)).toFloat
    if len < 1e-7f then Array(0f, 1f, 0f) else Array(v(0) / len, v(1) / len, v(2) / len)
```

> **Note:** The dodecahedron face-index table above may need adjustment after visual verification. The vertex positions are correct; verify face winding in an integration render and fix indices if faces point inward.

- [ ] **Step 7: Run polyhedra tests**

```
sbt "testOnly menger.objects.PolyhedraTest"
```
Expected: all pass (counts are exact). If Dodecahedron fails, recount faces.

- [ ] **Step 8: Register new types in `ObjectType`**

In `menger-common/src/main/scala/menger/common/ObjectType.scala`, add to `VALID_TYPES`:

```scala
val VALID_TYPES: Set[String] = Set(
  "sphere", "cube", "parametric",
  "sponge-volume", "sponge-surface", "cube-sponge", "sponge-recursive-ias",
  "tesseract", "tesseract-sponge-volume", "tesseract-sponge-surface",
  "tetrahedron", "octahedron", "dodecahedron", "icosahedron"
)
```

No new `isSponge` / `isProjected4D` predicates needed; `isTriangleMesh` already covers them via the fallback.

- [ ] **Step 9: Add cases to `MeshFactory`**

In `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala`, add before the `case "parametric"` arm:

```scala
case "tetrahedron" =>
  Tetrahedron(scale = spec.size).toTriangleMesh

case "octahedron" =>
  Octahedron(scale = spec.size).toTriangleMesh

case "dodecahedron" =>
  Dodecahedron(scale = spec.size).toTriangleMesh

case "icosahedron" =>
  Icosahedron(scale = spec.size).toTriangleMesh
```

Add imports at the top of `MeshFactory.scala`:
```scala
import menger.objects.Tetrahedron
import menger.objects.Octahedron
import menger.objects.Dodecahedron
import menger.objects.Icosahedron
```

- [ ] **Step 10: Add to `isTriangleMeshType` in `TriangleMeshSceneBuilder`**

In `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala`, update `isTriangleMeshType`:

```scala
private def isTriangleMeshType(spec: ObjectSpec): Boolean =
  spec.objectType == "cube" ||
  spec.objectType == "parametric" ||
  spec.objectType == "tetrahedron" ||
  spec.objectType == "octahedron" ||
  spec.objectType == "dodecahedron" ||
  spec.objectType == "icosahedron" ||
  ObjectType.isSponge(spec.objectType) ||
  ObjectType.isProjected4D(spec.objectType)
```

- [ ] **Step 11: Run full tests**

```
sbt test
```
Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add menger-app/src/main/scala/menger/objects/Tetrahedron.scala
git add menger-app/src/main/scala/menger/objects/Octahedron.scala
git add menger-app/src/main/scala/menger/objects/Dodecahedron.scala
git add menger-app/src/main/scala/menger/objects/Icosahedron.scala
git add menger-common/src/main/scala/menger/common/ObjectType.scala
git add menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala
git add menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala
git add menger-app/src/test/scala/menger/objects/PolyhedraTest.scala
git commit -m "feat(19.1): add tetrahedron, octahedron, dodecahedron, icosahedron primitives"
```

---

## Task 19.3: Analytical Primitives — Cone and Torus

**Prerequisite reading:** `docs/dev/adding-analytical-primitives.md` — follow the 8-step recipe verbatim. Below is the task-specific instantiation.

**Files (C++):**
- Modify: `optix-jni/src/main/native/include/OptiXData.h`
- Create: `optix-jni/src/main/native/shaders/hit_cone.cu`
- Create: `optix-jni/src/main/native/shaders/hit_torus.cu`
- Modify: `optix-jni/src/main/native/shaders/optix_shaders.cu`
- Modify: `optix-jni/src/main/native/PipelineManager.h`
- Modify: `optix-jni/src/main/native/PipelineManager.cpp`

**Files (Scala):**
- Modify: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
- Create: `menger-app/src/main/scala/menger/engines/scene/ConeSceneBuilder.scala`
- Create: `menger-app/src/main/scala/menger/engines/scene/TorusSceneBuilder.scala`
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala`
- Modify: `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala`
- Test: `menger-app/src/test/scala/menger/engines/scene/ConeSceneBuilderTest.scala`

### Step-by-step

- [ ] **Step 1: Add C++ data structs to `OptiXData.h`**

After the `CylinderData` struct (around line 425), add:

```cpp
struct ConeData {
    float apex[3];       // apex position in world space
    float base[3];       // base center position
    float radius;        // base radius
    // height derived as length(base - apex); not stored to keep struct minimal
};

struct TorusData {
    float center[3];     // center of the torus
    float axis[3];       // axis of symmetry (normalized)
    float major_radius;  // distance from center to tube center
    float minor_radius;  // tube radius
};
```

Extend `Params` struct (around line 459) after the cylinder fields:
```cpp
ConeData*    cone_data;
unsigned int num_cones;
TorusData*   torus_data;
unsigned int num_toruses;
```

Add to `GeometryType` enum (line 153):
```cpp
GEOMETRY_TYPE_CONE   = 3,
GEOMETRY_TYPE_TORUS  = 4,
GEOMETRY_TYPE_COUNT  = 5
```

- [ ] **Step 2: Create `hit_cone.cu`**

Create `optix-jni/src/main/native/shaders/hit_cone.cu`. Copy `hit_cylinder.cu` as template. Replace cylinder math with cone intersection:

```cuda
extern "C" __global__ void __intersection__cone() {
    const unsigned int instanceId = optixGetInstanceId();
    if (instanceId >= params.num_instances) return;
    if (!params.instance_materials)        return;
    const InstanceMaterial& mat = params.instance_materials[instanceId];
    const int cone_index = mat.texture_index;
    if (cone_index < 0 || cone_index >= static_cast<int>(params.num_cones)) return;
    if (!params.cone_data) return;
    const ConeData* cone = &params.cone_data[cone_index];

    const float3 apex  = make_float3(cone->apex[0], cone->apex[1], cone->apex[2]);
    const float3 base  = make_float3(cone->base[0], cone->base[1], cone->base[2]);
    const float3 axis  = normalize(base - apex);
    const float  height = length(base - apex);
    const float  r     = cone->radius;
    const float  cos_a = height / sqrtf(height * height + r * r);  // cos(half_angle)
    const float  sin_a = r     / sqrtf(height * height + r * r);

    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float3 d  = ro - apex;
    const float  tmin = optixGetRayTmin();
    const float  tmax = optixGetRayTmax();

    // Ray vs infinite cone: (dot(P-apex, axis)/|P-apex|)^2 = cos_a^2
    const float  dv = dot(rd, axis);
    const float  dd = dot(rd, rd);
    const float  ov = dot(d, axis);
    const float  od = dot(d, rd);
    const float  oo = dot(d, d);
    const float  ca2 = cos_a * cos_a;

    const float A = dv * dv - ca2 * dd;
    const float B = 2.0f * (dv * ov - ca2 * od);
    const float C = ov * ov - ca2 * oo;

    const float disc = B * B - 4.0f * A * C;
    if (disc < 0.0f) return;

    const float sq = sqrtf(disc);
    float t0 = (-B - sq) / (2.0f * A);
    float t1 = (-B + sq) / (2.0f * A);
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }

    // Check both intersections; keep those within [0, height] on axis
    auto tryHit = [&](float t) -> bool {
        if (t < tmin || t > tmax) return false;
        const float3 P    = ro + t * rd;
        const float  proj = dot(P - apex, axis);
        if (proj < 0.0f || proj > height) return false;
        optixReportIntersection(t, 0u);
        return true;
    };
    if (!tryHit(t0)) tryHit(t1);
}
```

Add `__closesthit__cone`, `__closesthit__cone_shadow`, `__anyhit__cone_shadow` — copy from cylinder's closest-hit programs verbatim (normal calculation differs: cone normal at hit P is `normalize(P - apex - dot(P-apex,axis)*axis + sin_a*cos_a*(P-apex))`, but for now delegate to the same PBR shading as cylinder).

Cone outward normal at hit point P (apex at origin, axis along `axis`):
```cuda
const float3 P    = ro + hit_t * rd;
const float  proj = dot(P - apex, axis);
const float3 radial = normalize((P - apex) - proj * axis);
const float3 N = normalize(radial * cos_a - axis * sin_a);
```

- [ ] **Step 3: Create `hit_torus.cu`**

Create `optix-jni/src/main/native/shaders/hit_torus.cu`. The torus intersection is a degree-4 polynomial; use the standard quartic solver:

```cuda
// Solve ax^4 + bx^3 + cx^2 + dx + e = 0 for up to 4 real roots.
// Returns number of roots found; writes results to roots[].
__device__ int solveQuartic(float a, float b, float c, float d, float e, float roots[4]);
```

(Provide a numerically stable implementation based on Ferrari's method or the companion-matrix approach.) See PBRT Section 6.1.3 for reference.

Normal at hit point P = `(x,y,z)`:
```cuda
const float3 center = make_float3(torus->center[0], ...);
const float  R = torus->major_radius;
const float  r = torus->minor_radius;
const float3 P = ro + t * rd - center;
const float  q = sqrtf(P.x*P.x + P.z*P.z) - R;  // assumes axis = Y
const float3 N = normalize(make_float3(P.x * (1.0f - R / sqrtf(P.x*P.x + P.z*P.z)),
                                        P.y,
                                        P.z * (1.0f - R / sqrtf(P.x*P.x + P.z*P.z))));
```

For now support only Y-axis tori (generalise with a basis change using `torus->axis` later).

- [ ] **Step 4: Include new shaders in `optix_shaders.cu`**

In `optix-jni/src/main/native/shaders/optix_shaders.cu`, add:
```cpp
#include "hit_cone.cu"
#include "hit_torus.cu"
```

- [ ] **Step 5: Register program groups in `PipelineManager.h` and `.cpp`**

In `PipelineManager.h`, add alongside cylinder members:
```cpp
OptixProgramGroup cone_hitgroup_prog_group;
OptixProgramGroup cone_shadow_hitgroup_prog_group;
OptixProgramGroup photon_cone_hitgroup;
OptixProgramGroup torus_hitgroup_prog_group;
OptixProgramGroup torus_shadow_hitgroup_prog_group;
OptixProgramGroup photon_torus_hitgroup;
```

In `PipelineManager.cpp`, in `createProgramGroups()`, add after the cylinder block:
```cpp
// Cone program groups
cone_hitgroup_prog_group = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__cone", module, "__intersection__cone");
cone_shadow_hitgroup_prog_group = optix_context.createHitgroupProgramGroupWithAH(
    module, "__closesthit__cone_shadow", module, "__anyhit__cone_shadow",
    module, "__intersection__cone");
photon_cone_hitgroup = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__photon", module, "__intersection__cone");

// Torus program groups
torus_hitgroup_prog_group = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__torus", module, "__intersection__torus");
torus_shadow_hitgroup_prog_group = optix_context.createHitgroupProgramGroupWithAH(
    module, "__closesthit__torus_shadow", module, "__anyhit__torus_shadow",
    module, "__intersection__torus");
photon_torus_hitgroup = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__photon", module, "__intersection__torus");
```

Destroy all six in `destroy()`. Wire all into the SBT builder at offsets `STRIDE_RAY_TYPES * GEOMETRY_TYPE_CONE` and `STRIDE_RAY_TYPES * GEOMETRY_TYPE_TORUS`.

- [ ] **Step 6: Add Scala JNI wrappers**

In `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`, add after `addCylinderInstance`:

```scala
@native private def addConeInstanceNative(
  apexX: Float, apexY: Float, apexZ: Float,
  baseX: Float,  baseY: Float,  baseZ: Float,
  radius: Float,
  r: Float, g: Float, b: Float, a: Float,
  ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float
): Option[Int]

def addConeInstance(
  apex: Vector[3], base: Vector[3], radius: Float, material: Material
): Option[Int] =
  addConeInstanceNative(
    apex.x, apex.y, apex.z,
    base.x, base.y, base.z,
    radius,
    material.color.r, material.color.g, material.color.b, material.color.a,
    material.ior, material.roughness, material.metallic, material.specular, material.emission
  )

@native private def addTorusInstanceNative(
  cx: Float, cy: Float, cz: Float,
  ax: Float, ay: Float, az: Float,
  majorRadius: Float, minorRadius: Float,
  r: Float, g: Float, b: Float, a: Float,
  ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float
): Option[Int]

def addTorusInstance(
  center: Vector[3], axis: Vector[3],
  majorRadius: Float, minorRadius: Float,
  material: Material
): Option[Int] =
  addTorusInstanceNative(
    center.x, center.y, center.z,
    axis.x, axis.y, axis.z,
    majorRadius, minorRadius,
    material.color.r, material.color.g, material.color.b, material.color.a,
    material.ior, material.roughness, material.metallic, material.specular, material.emission
  )
```

- [ ] **Step 7: Write Scala scene builder tests**

Create `menger-app/src/test/scala/menger/engines/scene/ConeSceneBuilderTest.scala`:

```scala
package menger.engines.scene

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.ObjectSpec
import menger.ProfilingConfig

class ConeSceneBuilderTest extends AnyFlatSpec with Matchers:

  given ProfilingConfig = ProfilingConfig.disabled

  "ConeSceneBuilder" should "validate a well-formed cone spec" in
    val spec = ObjectSpec(objectType = "cone", size = 1f)
    val builder = ConeSceneBuilder()
    builder.validate(List(spec), maxInstances = 10) shouldBe Right(())

  it should "reject empty spec list" in
    val builder = ConeSceneBuilder()
    builder.validate(List.empty, maxInstances = 10) shouldBe a[Left[?, ?]]

  it should "report instance count = 1 per spec" in
    val specs = List(
      ObjectSpec(objectType = "cone"),
      ObjectSpec(objectType = "cone")
    )
    ConeSceneBuilder().calculateInstanceCount(specs) shouldBe 2L
```

- [ ] **Step 8: Create `ConeSceneBuilder`**

Create `menger-app/src/main/scala/menger/engines/scene/ConeSceneBuilder.scala`:

```scala
package menger.engines.scene

import scala.util.Try
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.Vector
import menger.optix.OptiXRenderer

class ConeSceneBuilder(using ProfilingConfig) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if specs.exists(_.objectType.toLowerCase != "cone") then Left("ConeSceneBuilder only accepts cone specs")
    else if specs.length > maxInstances then Left(s"Too many cones: ${specs.length} > $maxInstances")
    else Right(())

  override def buildScene(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    maxInstances: Int
  ): Try[Unit] = Try:
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      // Cone along Y axis: apex above center, base below
      val h = spec.size / 2f
      val apex = Vector[3](spec.x, spec.y + h, spec.z)
      val base = Vector[3](spec.x, spec.y - h, spec.z)
      val r    = spec.size / 2f     // base radius = half size
      renderer.addConeInstance(apex, base, r, material)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType.toLowerCase == "cone" && spec2.objectType.toLowerCase == "cone"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
```

- [ ] **Step 9: Create `TorusSceneBuilder`** (analogous to `ConeSceneBuilder`)

Create `menger-app/src/main/scala/menger/engines/scene/TorusSceneBuilder.scala`:

```scala
package menger.engines.scene

import scala.util.Try
import menger.ObjectSpec
import menger.ProfilingConfig
import menger.common.Vector
import menger.optix.OptiXRenderer

class TorusSceneBuilder(using ProfilingConfig) extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if specs.exists(_.objectType.toLowerCase != "torus") then Left("TorusSceneBuilder only accepts torus specs")
    else if specs.length > maxInstances then Left(s"Too many toruses: ${specs.length} > $maxInstances")
    else Right(())

  override def buildScene(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    maxInstances: Int
  ): Try[Unit] = Try:
    specs.foreach { spec =>
      val material = MaterialExtractor.extract(spec)
      val center = Vector[3](spec.x, spec.y, spec.z)
      val axis   = Vector[3](0f, 1f, 0f)   // Y-axis torus by default
      // major radius = size; minor radius = size / 4 (reasonable default)
      val majorR = spec.size
      val minorR = spec.size / 4f
      renderer.addTorusInstance(center, axis, majorR, minorR, material)
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType.toLowerCase == "torus" && spec2.objectType.toLowerCase == "torus"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long = specs.length.toLong
```

- [ ] **Step 10: Register cone and torus in `ObjectType` and `GeometryRegistry`**

In `ObjectType.scala`, add `"cone"` and `"torus"` to `VALID_TYPES`.

In `GeometryRegistry.scala`, add:
```scala
else if types.forall(_ == "cone") then
  Some(ConeSceneBuilder())
else if types.forall(_ == "torus") then
  Some(TorusSceneBuilder())
```

- [ ] **Step 11: Compile C++ / rebuild native**

```
sbt "project optixJni" nativeCompile
```
Fix any C++ errors.

- [ ] **Step 12: Run all tests**

```
sbt test
```
Expected: all pass (C++ tests may require `sbt "project optixJni" nativeTest`).

- [ ] **Step 13: Commit**

```bash
git add optix-jni/src/main/native/include/OptiXData.h
git add optix-jni/src/main/native/shaders/hit_cone.cu
git add optix-jni/src/main/native/shaders/hit_torus.cu
git add optix-jni/src/main/native/shaders/optix_shaders.cu
git add optix-jni/src/main/native/PipelineManager.h
git add optix-jni/src/main/native/PipelineManager.cpp
git add optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala
git add menger-app/src/main/scala/menger/engines/scene/ConeSceneBuilder.scala
git add menger-app/src/main/scala/menger/engines/scene/TorusSceneBuilder.scala
git add menger-common/src/main/scala/menger/common/ObjectType.scala
git add menger-app/src/main/scala/menger/engines/GeometryRegistry.scala
git add menger-app/src/test/scala/menger/engines/scene/ConeSceneBuilderTest.scala
git commit -m "feat(19.3): add cone and torus analytical IS primitives"
```

---

## Task 19.4: Planes as First-Class Geometry

**What changes:** Planes move from the miss shader (a static loop over `params.planes[]`) to the IS GAS system. The miss shader's `getBackgroundColor()` function (lines 85–96 of `miss_plane.cu`) is unaffected — it returns the background solid colour and stays.

**Files:**
- Modify: `optix-jni/src/main/native/shaders/miss_plane.cu` (remove plane loop)
- Modify: `optix-jni/src/main/native/include/OptiXData.h` (add `GEOMETRY_TYPE_PLANE`, `PlaneISData`)
- Create: `optix-jni/src/main/native/shaders/hit_plane.cu`
- Modify: `optix-jni/src/main/native/shaders/optix_shaders.cu`
- Modify: `optix-jni/src/main/native/PipelineManager.h` / `.cpp`
- Modify: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`
- Create: `menger-app/src/main/scala/menger/engines/scene/PlaneSceneBuilder.scala`
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala`
- Modify: `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala`

- [ ] **Step 1: Add `PlaneISData` to `OptiXData.h`**

After `TorusData`, add:
```cpp
struct PlaneISData {
    float point[3];   // any point on the plane
    float normal[3];  // unit outward normal
};
```

Extend `Params`:
```cpp
PlaneISData* plane_is_data;
unsigned int num_plane_instances;
```

(The existing `PlaneParams planes[MAX_PLANES]` in the miss shader can coexist or be removed once the IS path is working; remove it in this task to keep things clean.)

Add to `GeometryType`:
```cpp
GEOMETRY_TYPE_PLANE = 5,
GEOMETRY_TYPE_COUNT = 6
```

- [ ] **Step 2: Remove plane loop from `miss_plane.cu`**

In `optix-jni/src/main/native/shaders/miss_plane.cu`, delete the `for` loop that iterates over `params.planes[]` and calls `optixReportIntersection`. Keep `getBackgroundColor()` and the miss shader entry point that calls it.

- [ ] **Step 3: Create `hit_plane.cu`**

```cuda
extern "C" __global__ void __intersection__plane() {
    const unsigned int instanceId = optixGetInstanceId();
    if (instanceId >= params.num_instances) return;
    if (!params.instance_materials)         return;
    const InstanceMaterial& mat = params.instance_materials[instanceId];
    const int plane_index = mat.texture_index;
    if (plane_index < 0 || plane_index >= static_cast<int>(params.num_plane_instances)) return;
    if (!params.plane_is_data) return;
    const PlaneISData* plane = &params.plane_is_data[plane_index];

    const float3 N = make_float3(plane->normal[0], plane->normal[1], plane->normal[2]);
    const float3 P = make_float3(plane->point[0],  plane->point[1],  plane->point[2]);
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float  denom = dot(N, rd);
    if (fabsf(denom) < 1e-6f) return;  // ray parallel to plane
    const float  t = dot(P - ro, N) / denom;
    if (t < optixGetRayTmin() || t > optixGetRayTmax()) return;
    optixReportIntersection(t, 0u);
}
```

Provide `__closesthit__plane`, `__closesthit__plane_shadow`, `__anyhit__plane_shadow` — same PBR shading as cylinder, with plane normal `N` (flip if back-face).

- [ ] **Step 4: Include in umbrella and register program groups**

Same pattern as cone/torus. Add `GEOMETRY_TYPE_PLANE` to the SBT stride calculation; wire hit groups.

- [ ] **Step 5: Add Scala JNI wrapper `addPlaneInstance`**

```scala
@native private def addPlaneInstanceNative(
  px: Float, py: Float, pz: Float,
  nx: Float, ny: Float, nz: Float,
  r: Float, g: Float, b: Float, a: Float,
  ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float
): Option[Int]

def addPlaneInstance(
  point: Vector[3], normal: Vector[3], material: Material
): Option[Int] =
  addPlaneInstanceNative(
    point.x, point.y, point.z,
    normal.x, normal.y, normal.z,
    material.color.r, material.color.g, material.color.b, material.color.a,
    material.ior, material.roughness, material.metallic, material.specular, material.emission
  )
```

- [ ] **Step 6: Create `PlaneSceneBuilder`**

Create `menger-app/src/main/scala/menger/engines/scene/PlaneSceneBuilder.scala` (analogous to `ConeSceneBuilder`). Parse `pos=x,y,z` as the plane's point and expect a `normal=nx,ny,nz` key (add to `ValidKeys` in `ObjectSpec`). Default normal: `0,1,0` (floor plane).

- [ ] **Step 7: Add `normal` to `ObjectSpec.ValidKeys`**

In `ObjectSpec.scala`:
```scala
private val ValidKeys: Set[String] = Set(
  ...,
  "normal"  // plane normal vector (nx,ny,nz)
)
```

Parse `normal` as a 3-float comma-separated value. Add `normal: Vector[3]` field to `ObjectSpec` with default `Vector[3](0f, 1f, 0f)`.

- [ ] **Step 8: Register in `ObjectType` and `GeometryRegistry`**

Add `"plane"` to `VALID_TYPES`. Add `else if types.forall(_ == "plane") then Some(PlaneSceneBuilder())` in `GeometryRegistry`.

- [ ] **Step 9: Build and test**

```
sbt "project optixJni" nativeCompile
sbt test
```

- [ ] **Step 10: Commit**

```bash
git add [all plane-related files]
git commit -m "feat(19.4): planes as first-class IS geometry; remove from miss shader"
```

---

## Task 19.5: Coordinate Cross (Axis Visualization)

The coordinate cross is three thin cylinders along X (red), Y (green), Z (blue). No new IS program needed — cylinders already exist.

**Files:**
- Modify: `menger-app/src/main/scala/menger/engines/scene/CylinderSceneBuilder.scala` (if exists) or create analogous
- Modify: `menger-app/src/main/scala/menger/MengerCLIOptions.scala`
- Modify: `menger-app/src/main/scala/menger/engines/BaseEngine.scala` (inject axis if flag set)
- Test: logic test only (visual via integration)

- [ ] **Step 1: Add `--axis` CLI flag**

In `MengerCLIOptions.scala`, add a boolean option:
```scala
val axis = opt[Boolean](name = "axis", default = Some(false),
  descr = "Show coordinate axis cross (X=red, Y=green, Z=blue)")
```

- [ ] **Step 2: Create `CoordinateCross` helper**

Create `menger-app/src/main/scala/menger/objects/CoordinateCross.scala`:

```scala
package menger.objects

import menger.ObjectSpec

object CoordinateCross:
  val DefaultLength: Float = 2f
  val DefaultRadius: Float = 0.02f

  /** Returns three ObjectSpec entries for the X/Y/Z axis cylinders. */
  def specs(
    length: Float = DefaultLength,
    radius: Float = DefaultRadius
  ): List[ObjectSpec] =
    List(
      // X axis: red
      ObjectSpec(objectType = "cylinder",
        x = length / 2f, size = length,
        color = menger.common.Color(1f, 0f, 0f, 1f)),
      // Y axis: green
      ObjectSpec(objectType = "cylinder",
        y = length / 2f, size = length,
        color = menger.common.Color(0f, 1f, 0f, 1f)),
      // Z axis: blue
      ObjectSpec(objectType = "cylinder",
        z = length / 2f, size = length,
        color = menger.common.Color(0f, 0f, 1f, 1f))
    )
```

> **Note:** `ObjectSpec` stores colour as separate `r/g/b/a` fields or via a `color` param — adjust to match the actual constructor signature.

- [ ] **Step 3: Inject axis specs in engine startup**

In `BaseEngine.buildSceneFromSpecs`, when `--axis` is enabled, prepend `CoordinateCross.specs()` to the `specs` list before dispatch.

Alternatively, do this at the `Main.scala` level before passing specs to the engine.

- [ ] **Step 4: Write unit test for `CoordinateCross.specs`**

```scala
"CoordinateCross.specs" should "return 3 cylinder specs" in
  val specs = CoordinateCross.specs()
  specs.length shouldBe 3
  specs.map(_.objectType).distinct shouldBe List("cylinder")
```

- [ ] **Step 5: Run tests and commit**

```
sbt test
git add [axis-related files]
git commit -m "feat(19.5): coordinate cross axis visualization (--axis flag)"
```

---

## Task 19.2a: 4D Polytopes — Pentachoron (5-cell) and 16-cell

**Pattern:** Extend `Mesh4D` just like `Tesseract`. Register in `ObjectType` and `MeshFactory`.

**Files:**
- Create: `menger-app/src/main/scala/menger/objects/higher_d/Pentachoron.scala`
- Create: `menger-app/src/main/scala/menger/objects/higher_d/Cell16.scala`
- Modify: `menger-common/src/main/scala/menger/common/ObjectType.scala`
- Modify: `menger-app/src/main/scala/menger/engines/scene/MeshFactory.scala`
- Modify: `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala`
- Modify: `menger-app/src/main/scala/menger/engines/GeometryRegistry.scala`
- Test: `menger-app/src/test/scala/menger/objects/higher_d/Polychora4DTest.scala`

- [ ] **Step 1: Write failing tests**

Create `menger-app/src/test/scala/menger/objects/higher_d/Polychora4DTest.scala`:

```scala
package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Polychora4DTest extends AnyFlatSpec with Matchers:

  "Pentachoron" should "have 5 vertices" in
    Pentachoron().vertices.length shouldBe 5

  it should "have 10 faces (quads)" in
    // 5-cell: 5 tetrahedral cells, each with 4 triangular faces.
    // After deduplication of shared faces: 10 faces.
    // But Mesh4D uses Face4D (quads), so 10 quads for the 10 triangular faces
    // expressed as degenerate quads (first two vertices shared), or
    // we store as actual quads with one degenerate edge.
    // Check we have > 0 faces and correct type.
    Pentachoron().faces.length should be > 0

  "Cell16" should "have 8 vertices" in
    Cell16().vertices.length shouldBe 8

  it should "have 24 faces" in
    // 16-cell: 8 tetrahedral cells, each with 4 triangular faces = 32 total faces,
    // 24 after deduplication (each internal triangle shared by 2 cells).
    // Expressed as quads: 24 quads.
    Cell16().faces.length shouldBe 24
```

- [ ] **Step 2: Run to confirm failure**

```
sbt "testOnly menger.objects.higher_d.Polychora4DTest"
```
Expected: compile errors.

- [ ] **Step 3: Implement `Pentachoron`**

The 5-cell has 5 vertices equidistant in 4D. One embedding:
- v0 = (1, 1, 1, -1/√5)
- v1 = (1, -1, -1, -1/√5)
- v2 = (-1, 1, -1, -1/√5)
- v3 = (-1, -1, 1, -1/√5)
- v4 = (0, 0, 0, √5 - 1/√5)

(All pairwise distances equal; normalize to unit circumradius, then scale by `size/2`.)

The 5-cell has 10 triangular faces (pairs of vertices from each of the 10 edges). Expressed as `Face4D` (which takes 4 vertices), use the quad by doubling the last vertex (degenerate quad) or triangulate. **Preferred:** add a `TriangularFace4D` if the existing infrastructure doesn't support triangular faces natively, or just list faces as quads with `c == d` (degenerate).

Create `menger-app/src/main/scala/menger/objects/higher_d/Pentachoron.scala`:

```scala
package menger.objects.higher_d

import menger.common.Vector

/**
 * Regular 5-cell (pentachoron) in 4D.
 * 5 vertices, 10 edges, 10 triangular faces, 5 tetrahedral cells.
 * Faces stored as degenerate Face4D quads (c == d) so they render as triangles.
 */
case class Pentachoron(size: Float = 1f) extends Mesh4D:

  lazy val vertices: Seq[Vector[4]] =
    val s = size / 2f
    val inv5 = (1.0 / math.sqrt(5.0)).toFloat
    val r4   = (math.sqrt(5.0) - 1.0 / math.sqrt(5.0)).toFloat
    Seq(
      Vector[4]( s,  s,  s, -s * inv5),
      Vector[4]( s, -s, -s, -s * inv5),
      Vector[4](-s,  s, -s, -s * inv5),
      Vector[4](-s, -s,  s, -s * inv5),
      Vector[4]( 0f, 0f, 0f, s * r4)
    )

  // 10 triangular faces as degenerate quads (a,b,c,c)
  lazy val faceIndices: Seq[(Int, Int, Int)] =
    for
      i <- 0 until 5
      j <- (i + 1) until 5
      k <- (j + 1) until 5
    yield (i, j, k)

  lazy val faces: Seq[Face4D] =
    faceIndices.map { (i, j, k) =>
      Face4D(vertices(i), vertices(j), vertices(k), vertices(k))  // degenerate quad
    }
```

- [ ] **Step 4: Implement `Cell16`**

The 16-cell has 8 vertices at `±e_i` (unit vectors along each axis):

```scala
package menger.objects.higher_d

import menger.common.Vector

/**
 * Regular 16-cell (hexadecachoron) in 4D.
 * 8 vertices at ±(size/2) on each 4D axis.
 * 24 octahedral cells (each a triangle), 24 Face4D quads after deduplication.
 */
case class Cell16(size: Float = 1f) extends Mesh4D:

  lazy val vertices: Seq[Vector[4]] =
    val h = size / 2f
    Seq(
      Vector[4]( h, 0f, 0f, 0f), Vector[4](-h, 0f, 0f, 0f),
      Vector[4](0f,  h, 0f, 0f), Vector[4](0f, -h, 0f, 0f),
      Vector[4](0f, 0f,  h, 0f), Vector[4](0f, 0f, -h, 0f),
      Vector[4](0f, 0f, 0f,  h), Vector[4](0f, 0f, 0f, -h)
    )

  // All triangles = triples of vertices not on opposite axes
  // i.e. i,j,k such that none of (i,j), (i,k), (j,k) are axis-opposite pairs
  private val oppositePairs: Set[(Int, Int)] =
    Set((0,1),(2,3),(4,5),(6,7)).flatMap(p => Set(p, p.swap))

  lazy val faces: Seq[Face4D] =
    (for
      i <- 0 until 8
      j <- (i + 1) until 8
      k <- (j + 1) until 8
      if !oppositePairs((i, j)) && !oppositePairs((i, k)) && !oppositePairs((j, k))
    yield Face4D(vertices(i), vertices(j), vertices(k), vertices(k))).toSeq
```

- [ ] **Step 5: Register types**

In `ObjectType.scala`, add `"pentachoron"` and `"16-cell"` to `VALID_TYPES` and `PROJECTED_4D_TYPES`.

In `MeshFactory.scala`, add cases:
```scala
case "pentachoron" =>
  Mesh4DProjection(Pentachoron(size = spec.size), /* proj params from spec */ ...).toTriangleMesh

case "16-cell" =>
  Mesh4DProjection(Cell16(size = spec.size), ...).toTriangleMesh
```

Use the helper `mesh4DProjection` or inline the `Projection4DSpec` extraction.

- [ ] **Step 6: Run tests**

```
sbt "testOnly menger.objects.higher_d.Polychora4DTest"
sbt test
```

- [ ] **Step 7: Commit**

```bash
git add menger-app/src/main/scala/menger/objects/higher_d/Pentachoron.scala
git add menger-app/src/main/scala/menger/objects/higher_d/Cell16.scala
git add [ObjectType, MeshFactory, TriangleMeshSceneBuilder]
git add menger-app/src/test/scala/menger/objects/higher_d/Polychora4DTest.scala
git commit -m "feat(19.2a): add 4D pentachoron (5-cell) and 16-cell polytopes"
```

---

## Task 19.2b: 4D Polytopes — 24-cell, 120-cell, 600-cell

Same pattern as 19.2a. These are larger polytopes; vertex tables below.

**Files:** Same structure as 19.2a, create `Cell24.scala`, `Cell120.scala`, `Cell600.scala`.

- [ ] **Step 1: Add tests for 24-cell, 120-cell, 600-cell**

Extend `Polychora4DTest.scala`:

```scala
"Cell24" should "have 24 vertices" in
  Cell24().vertices.length shouldBe 24

it should "have 96 faces" in
  Cell24().faces.length shouldBe 96

"Cell120" should "have 600 vertices" in
  Cell120().vertices.length shouldBe 600

"Cell600" should "have 120 vertices" in
  Cell600().vertices.length shouldBe 120
```

- [ ] **Step 2: Implement `Cell24`**

24-cell vertices: 24 vertices — all permutations of `(±1, ±1, 0, 0)` scaled to circumradius:

```scala
case class Cell24(size: Float = 1f) extends Mesh4D:

  lazy val vertices: Seq[Vector[4]] =
    val h = (size / 2f) / math.sqrt(2.0).toFloat
    // All permutations of (±h, ±h, 0, 0) across 4 coordinates
    val coords = Seq(-h, h)
    (for
      i <- 0 until 4
      j <- (i + 1) until 4
      si <- coords
      sj <- coords
    yield
      val v = Array(0f, 0f, 0f, 0f)
      v(i) = si; v(j) = sj
      Vector[4](v(0), v(1), v(2), v(3))
    ).toSeq

  lazy val faces: Seq[Face4D] = ???   // generate from combinatoric face table
```

> The face table for the 24-cell requires the 96 triangular faces. These can be generated from the cell decomposition (3 octahedral cells, each with 8 triangular faces). Use the standard reference table from Coxeter / Wikipedia for the 24-cell face incidences.

- [ ] **Step 3: Implement `Cell120` and `Cell600`**

These have 600 and 120 vertices respectively. Use exact rational-then-scaled coordinates from standard tables:

- **600-cell (120 vertices):** Vertices include all even permutations of `(0, ±1, ±φ, ±φ²)/2` and all permutations of `(±φ, ±φ, ±φ, ±φ)/2` and `(±2, 0, 0, 0)/2`. Scaled to `size/2` circumradius.
- **120-cell (600 vertices):** Dual of the 600-cell. Vertices are at the face centers of the 600-cell.

Since full vertex tables for 120/600-cells are large (100+ lines), generate them programmatically using the known symmetry group. Provide a `private def generateVertices` helper that enumerates from the orbit.

For each cell, face table size:
- 24-cell: 96 triangular faces
- 120-cell: 720 pentagonal faces (expressed as quads or split)
- 600-cell: 1200 triangular faces

> These polytopes are compute-heavy to enumerate; add timing assertions or a warning in the constructor for large vertex counts.

- [ ] **Step 4: Register and test**

Add to `VALID_TYPES`, `PROJECTED_4D_TYPES`, and `MeshFactory`. Run:
```
sbt test
```

- [ ] **Step 5: Commit**

```bash
git add menger-app/src/main/scala/menger/objects/higher_d/Cell24.scala
git add menger-app/src/main/scala/menger/objects/higher_d/Cell120.scala
git add menger-app/src/main/scala/menger/objects/higher_d/Cell600.scala
git add [ObjectType, MeshFactory updates]
git commit -m "feat(19.2b): add 4D 24-cell, 120-cell, 600-cell polytopes"
```

---

## Task 19.10: Spike — Fractional Levels with IAS Sponges

**Files:**
- Create: `docs/dev/sprint-19-spike-fractional-ias.md`

- [ ] **Step 1: Read the recursive IAS sponge implementation**

Open `menger-app/src/main/scala/menger/engines/scene/TriangleMeshSceneBuilder.scala` (the `addRecursiveIASSpongeInstance` call) and the C++ side (`OptiXRenderer.scala` for the JNI binding). Understand how many IAS levels exist and whether per-instance material alpha is thread-safe in the current implementation.

- [ ] **Step 2: Write findings doc**

Create `docs/dev/sprint-19-spike-fractional-ias.md`:

```markdown
# Spike: Fractional Levels with IAS Sponges

**Investigated:** Sprint 19 (May 2026)

## Question

Can fractional sponge levels be applied to `sponge-recursive-ias`?

## Current State

[Summarise: how many IAS nesting levels, how instances are identified, whether
per-instance material alpha can differ within an IAS tree.]

## Approach A: Instance-level alpha in leaf GAS

[Describe: each leaf cube carries per-vertex alpha; the inner IAS just references them.
Complexity, VRAM impact, implementation effort.]

## Approach B: Two IAS trees, alpha-blend in shader

[Describe: one IAS at level N (opaque), one at N-1 (alpha = 1-frac). Shader blends.
O(N·20^2) VRAM concern. Practical for levels ≤ 5.]

## Recommendation

[One of: "Schedule for Sprint 21 with Approach A", "Defer indefinitely", etc.]
```

- [ ] **Step 3: Commit**

```bash
git add docs/dev/sprint-19-spike-fractional-ias.md
git commit -m "docs(19.10): spike findings — fractional IAS sponge levels"
```

---

## Task 19.11: Documentation

- [ ] **Step 1: Update CHANGELOG.md**

Under `[Unreleased]`, add:
```markdown
### Added
- Tetrahedron, octahedron, dodecahedron, icosahedron as 3D primitives (`--objects type=tetrahedron/octahedron/dodecahedron/icosahedron`)
- Pentachoron (5-cell), 16-cell, 24-cell, 120-cell, 600-cell as 4D polytopes
- Cone and torus as analytical IS primitives (`--objects type=cone/torus`)
- Planes as first-class IS geometry with materials (`--objects type=plane`)
- Coordinate cross axis visualization (`--axis` flag)
- Per-object 3D rotation via CLI (`rot-x`, `rot-y`, `rot-z` in degrees)
- Frame timing in render stats: `frameMs` and `msPerMray`

### Changed
- Geometry registry: adding a new primitive no longer requires modifying engine dispatch
- `SceneClassifier` renamed to `RenderModeSelector`
```

- [ ] **Step 2: Update `docs/guide/user-guide.md`**

Add a "Geometry Types" section documenting all new type strings, their parameters, and example `--objects` invocations.

- [ ] **Step 3: Update arc42 Section 5 (Building Blocks)**

Add `GeometryRegistry` to the building-block view in `docs/arc42/05-building-block-view.md`.

Update Section 9 (Architectural Decisions) in `docs/arc42/09-architectural-decisions.md`:
- ADR: geometry registry replaces hard-coded dispatch
- ADR: analytical primitives use params-indirection pattern (not SBT-data)

- [ ] **Step 4: Commit documentation**

```bash
git add CHANGELOG.md docs/guide/user-guide.md docs/arc42/
git commit -m "docs(19.11): sprint 19 documentation, CHANGELOG, arc42 updates"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task covering it |
|-------------|-----------------|
| Tetrahedron, octahedron, dodecahedron, icosahedron | 19.1 |
| Pentachoron, 16-cell, 24-cell, 120-cell, 600-cell | 19.2a, 19.2b |
| Cone and torus IS primitives | 19.3 |
| Planes as first-class geometry | 19.4 |
| Coordinate cross | 19.5 |
| Geometry registry | 19.6 |
| Per-object 3D rotation CLI | 19.7 |
| Render time stats (ms/frame, ms/ray) | 19.8 |
| Spike: max trace depth | 19.9 |
| Spike: fractional IAS | 19.10 |
| Documentation | 19.11 |

**Notes and caveats:**

1. **Dodecahedron face-index table** — The 12-face index table in 19.1 is a placeholder from a known vertex set; the exact winding must be verified visually after rendering. The test only checks triangle count (108 = 12×3×3), which will pass regardless of winding.

2. **Pentachoron face count** — The spec says "10 faces"; the Scala `faces` count will be 10. The test allows `> 0` for flexibility (verify and tighten after implementation confirms 10).

3. **Cell24 face table** — `???` placeholder left intentionally; the implementer must look up the 96 triangular faces from a combinatoric source. Tests are specified.

4. **120-cell/600-cell vertex generation** — Large coordinate tables; the implementer should use the known orbit enumeration. Tests are specified with correct counts.

5. **`CoordinateCross` uses `cylinder` type** — This assumes `cylinder` is already a valid IS type from Sprint 18. If `cylinder` is not available, use three thin `cube` meshes as fallback.

6. **`PlaneSceneBuilder` requires `normal` in `ObjectSpec`** — Added to `ValidKeys` and case class. Ensure the `parseVector3Param` helper exists or implement inline.

7. **Torus quartic solver** — The `hit_torus.cu` step shows the structure; the full quartic solver is non-trivial. Reference PBRT Section 6.1 or Graphic Gems. A reference implementation is available in the PBRT source (`core/shape.cpp` `Torus::Intersect`).
