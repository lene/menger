# Task 15.2: Parametric Surfaces f(u,v) — Design Spec

**Date:** 2026-03-25
**Sprint:** 15
**Estimate:** 4h
**Prerequisites:** Sprint 14 (triangle mesh pipeline, IAS scene builder)

---

## Goal

Add a general parametric surface infrastructure to the Scala DSL. Users define a surface as
a function `(u: Float, v: Float) => Vec3`, which gets tessellated into a triangle mesh and
rendered via the existing triangle mesh pipeline. No C++ or shader changes required.

This is a prerequisite for Sprint 18.3 (DSL primitives: cone, torus) and Sprint 20.2
(parametric surfaces in 4D).

---

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| OptiX approach | Tessellation to triangles | OptiX has no native parametric surface type. Custom intersection programs exist but require per-shape analytical math. Tessellation uses HW-accelerated ray-triangle intersection. |
| Analytical intersection for known shapes | Backlog (low priority) | YAGNI — demonstrate concrete need first. Cylinder/cone/torus via custom intersection programs can be added later. |
| Open vs closed surfaces | `closed` flag for seam welding only | No IOR restriction. Trust the user. IOR on open/non-orientable surfaces may produce interesting artifacts. |
| Specification method | DSL-only | Arbitrary Scala lambdas `(Float, Float) => Vec3`. No CLI `--objects type=parametric` support — lambdas can't be expressed as CLI strings. Named presets can be added as example DSL scenes. |
| Normal computation | Numerical partial derivatives | Finite differences: `cross(df/du, df/dv)`. Accurate at typical resolutions. No user-supplied normal function. |
| Default resolution | uSteps = 64, vSteps = 32 | ~4K triangles. Asymmetric because most surfaces have more variation in u. |
| Memory warning | Log warning when uSteps x vSteps > 1,000,000 | ~170 MB with BVH at 1M vertices. No hard limit. |

---

## Architecture

The parametric surface is a DSL-layer concept that produces `TriangleMeshData` — the same
type used by all existing mesh objects (cube, sponge, tesseract). No changes needed in the
OptiX JNI layer or C++ shaders.

```
DSL ParametricSurface
    -> ParametricTessellator.tessellate(f, uRange, vRange, uSteps, vSteps, closedU, closedV)
    -> TriangleMeshData(vertices, indices, vertexStride = 8)
    -> existing TriangleMeshSceneBuilder pipeline
    -> OptiXRenderer.setTriangleMesh / addTriangleMeshInstance
```

---

## DSL API

```scala
case class ParametricSurface(
  f: (Float, Float) => Vec3,
  uRange: (Float, Float) = (0f, 2f * math.Pi.toFloat),
  vRange: (Float, Float) = (0f, math.Pi.toFloat),
  uSteps: Int = 64,
  vSteps: Int = 32,
  closedU: Boolean = false,      // Weld u-seam (first/last column share vertices)
  closedV: Boolean = false,      // Weld v-seam (first/last row share vertices)
  pos: Vec3 = Vec3(0, 0, 0),
  size: Float = 1.0f,
  ior: Float = 1.0f,
  material: Option[Material] = None,
  color: Option[Color] = None,
  texture: Option[String] = None, // UV coords generated; texture mapping works for free
  rotation: Vec3 = Vec3.Zero
) extends SceneObject
```

Note: `closed` is split into `closedU` and `closedV` because some surfaces wrap in one
parameter but not the other (e.g., a cylinder wraps in u but is open in v).

---

## Tessellation Algorithm

`ParametricTessellator` (new object in `menger-app/src/main/scala/menger/objects/`):

### Vertex Generation

1. Sample `f(u, v)` on a regular grid of `(uSteps + 1) x (vSteps + 1)` points.
   - `u_i = uMin + i * (uMax - uMin) / uSteps` for `i in 0..uSteps`
   - `v_j = vMin + j * (vMax - vMin) / vSteps` for `j in 0..vSteps`

2. For each vertex, compute the normal via central finite differences:
   ```
   epsilon = 1e-4 * max(|uMax - uMin|, |vMax - vMin|)
   du = (f(u + epsilon, v) - f(u - epsilon, v)) / (2 * epsilon)
   dv = (f(u, v + epsilon) - f(u, v - epsilon)) / (2 * epsilon)
   normal = normalize(cross(du, dv))
   ```

3. UV texture coordinates normalized to [0, 1]:
   ```
   texU = (u - uMin) / (uMax - uMin)
   texV = (v - vMin) / (vMax - vMin)
   ```

4. **Degenerate normal fallback:** when `length(cross(du, dv)) < 1e-8`, the surface
   parameterization is singular (e.g., at poles of a sphere). Fall back to using the
   normalized position vector as the normal, or `(0, 1, 0)` if position is also near zero.

5. Pack each vertex as 8 floats: `[px, py, pz, nx, ny, nz, texU, texV]`
   This matches `TriangleMeshData` with `vertexStride = 8`.

### Seam Welding (closed surfaces)

If `closedU = true`: when generating indices, the last column (`i = uSteps`) references
the same vertex indices as the first column (`i = 0`) instead of using separate vertices.

If `closedV = true`: same for last row (`j = vSteps`) referencing first row (`j = 0`).

This eliminates the visible seam at the wrap-around boundary.

### Index Generation

For each grid cell `(i, j)` where `i in 0..uSteps-1` and `j in 0..vSteps-1`:
```
topLeft     = vertexIndex(i,     j)
topRight    = vertexIndex(i + 1, j)
bottomLeft  = vertexIndex(i,     j + 1)
bottomRight = vertexIndex(i + 1, j + 1)

Triangle 1: (topLeft, bottomLeft, bottomRight)
Triangle 2: (topLeft, bottomRight, topRight)
```

Where `vertexIndex(i, j)` accounts for seam welding (maps `i = uSteps` to `i = 0`
when `closedU`, etc.).

### Memory Warning

Before tessellation, check `uSteps * vSteps > 1_000_000`. If exceeded, log a warning:
```
WARN: Parametric surface tessellation is very high resolution (uSteps x vSteps = N vertices).
      This will use approximately M MB of GPU memory. Consider reducing resolution.
```

---

## Scene Builder Integration

### ObjectSpec Extension

Add an optional `meshData: Option[TriangleMeshData]` field to `ObjectSpec`. When a
`ParametricSurface` converts to `ObjectSpec` via `toObjectSpec`, it eagerly tessellates
the surface and stores the result in this field.

```scala
// In ParametricSurface.toObjectSpec — follow same conversion pattern as Cube.toObjectSpec:
val mesh = ParametricTessellator.tessellate(f, uRange, vRange, uSteps, vSteps, closedU, closedV)
ObjectSpec(
  objectType = "parametric",
  x = pos.x, y = pos.y, z = pos.z,
  size = size,
  color = color.map(_.toCommonColor),
  ior = material.map(_.ior).getOrElse(ior),
  material = material.map(_.toOptixMaterial),
  texture = texture,
  rotX = rotation.x, rotY = rotation.y, rotZ = rotation.z,
  meshData = Some(mesh)
)
```

### MeshFactory

Add a `"parametric"` case to `MeshFactory.create()`:
```scala
case "parametric" =>
  spec.meshData.getOrElse(
    throw IllegalStateException("Parametric surface missing pre-tessellated mesh data")
  )
```

### ObjectType

Add `"parametric"` to `ObjectType.VALID_TYPES` and ensure `isTriangleMeshType("parametric")`
returns `true`. This routes parametric surfaces through `TriangleMeshSceneBuilder`
automatically via `SceneClassifier`.

**Note:** `TriangleMeshSceneBuilder` has its own private `isTriangleMeshType` method
(independent of `SceneClassifier`) that must also be updated to accept `"parametric"`,
otherwise `validate()` will reject parametric surfaces.

### SceneConverter

`SceneConverter.validateSceneMaterials` uses exhaustive pattern matching on `SceneObject`.
Adding `ParametricSurface extends SceneObject` requires a new case in the match:
```scala
case obj: ParametricSurface => obj.material.foreach(warnMaterial)
```

---

## Example DSL Scenes

All example scenes follow the existing pattern: a Scala `object` with `val scene = Scene(...)`,
registered via `SceneRegistry.register(...)`. See `examples.dsl.GlassSphere` for reference.

### Sphere (compare to built-in)

```scala
object ParametricSphere:
  import scala.math.*
  val TwoPi = 2f * Pi.toFloat

  val scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(cos(u) * sin(v), cos(v), sin(u) * sin(v)),
      uRange = (0f, TwoPi), vRange = (0f, TwoPi.toFloat / 2f),
      closedU = true, closedV = false,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-sphere", scene)
```

### Torus (closed, glass)

```scala
object ParametricTorus:
  import scala.math.*
  val (R, r) = (1.0f, 0.4f)
  val TwoPi = 2f * Pi.toFloat

  val scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(
        (R + r * cos(v)) * cos(u),
        r * sin(v),
        (R + r * cos(v)) * sin(u)),
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      closedU = true, closedV = true,
      material = Some(Material.Glass)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-torus", scene)
```

### Wavy Sheet (open, IOR experiment)

```scala
object ParametricWavySheet:
  import scala.math.*

  val scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => Vec3(u, 0.3f * sin(u * 2).toFloat * cos(v * 2).toFloat, v),
      uRange = (-2f, 2f), vRange = (-2f, 2f),
      uSteps = 64, vSteps = 64,
      closedU = false, closedV = false,
      ior = 1.5f
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
    planes = List(Plane(Y at -1.5))
  )
  SceneRegistry.register("parametric-wavy-sheet", scene)
```

### Moebius Strip (open, film material)

```scala
object ParametricMoebius:
  import scala.math.*
  val TwoPi = 2f * Pi.toFloat

  val scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => {
        val halfU = u / 2f
        val w = v - 0.5f  // v in [-0.5, 0.5] for strip width
        Vec3(
          (1f + w * cos(halfU).toFloat) * cos(u).toFloat,
          (1f + w * cos(halfU).toFloat) * sin(u).toFloat,
          w * sin(halfU).toFloat)
      },
      uRange = (0f, TwoPi), vRange = (0f, 1f),
      uSteps = 128, vSteps = 16,
      closedU = false, closedV = false,  // Cannot close — half-twist
      material = Some(Material.Film)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-moebius", scene)
```

### Figure-8 Klein Bottle (non-orientable, IOR + film variants)

```scala
object ParametricKleinBottle:
  import scala.math.*
  val TwoPi = 2f * Pi.toFloat
  val a = 2.0f

  val scene = Scene(
    objects = List(ParametricSurface(
      f = (u, v) => {
        val cosU = cos(u).toFloat; val sinU = sin(u).toFloat
        val cosHalfU = cos(u / 2f).toFloat; val sinHalfU = sin(u / 2f).toFloat
        val sinV = sin(v).toFloat; val sin2V = sin(2f * v).toFloat
        val r = a + cosHalfU * sinV - sinHalfU * sin2V
        Vec3(r * cosU, r * sinU, sinHalfU * sinV + cosHalfU * sin2V)
      },
      uRange = (0f, TwoPi), vRange = (0f, TwoPi),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      ior = 1.5f  // Non-orientable + IOR: expect interesting artifacts
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-klein-bottle", scene)

object ParametricKleinBottleFilm:
  // Same geometry as ParametricKleinBottle but with film material
  val scene = Scene(
    objects = List(ParametricSurface(
      f = ParametricKleinBottle.scene.objects.head.asInstanceOf[ParametricSurface].f,
      uRange = (0f, 2f * math.Pi.toFloat), vRange = (0f, 2f * math.Pi.toFloat),
      uSteps = 128, vSteps = 64,
      closedU = true, closedV = true,
      material = Some(Material.Film)
    )),
    lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f))
  )
  SceneRegistry.register("parametric-klein-bottle-film", scene)
```

---

## Interactive Manual Test Cases

Add to `scripts/manual-test.sh` interactive tests array:

- `Parametric torus (glass)` — `--scene examples.dsl.ParametricTorus`
- `Parametric wavy sheet (IOR on open surface)` — `--scene examples.dsl.ParametricWavySheet`
- `Parametric Moebius strip (film material)` — `--scene examples.dsl.ParametricMoebius`
- `Parametric Klein bottle (IOR, non-orientable)` — `--scene examples.dsl.ParametricKleinBottle`
- `Parametric Klein bottle (film material)` — `--scene examples.dsl.ParametricKleinBottleFilm`
- `Parametric sphere (compare to built-in)` — `--scene examples.dsl.ParametricSphere`

---

## Unit Tests (`ParametricTessellatorSuite`)

1. **Flat plane** — `(u, v) => Vec3(u, 0, v)`: correct vertex count `(uSteps+1)*(vSteps+1)`,
   correct triangle count `2*uSteps*vSteps`, all normals approximately `(0, 1, 0)` or `(0, -1, 0)`
2. **Closed sphere** — closedU + closedV: first/last column share indices, vertex count is
   `uSteps * vSteps` (reduced from `(uSteps+1)*(vSteps+1)` by seam welding)
3. **Open surface** — closedU = false: first and last column have distinct vertex indices,
   vertex count is `(uSteps+1)*(vSteps+1)`
4. **Normal accuracy** — tessellated sphere normals vs analytical `normalize(position)`:
   max angular error < some threshold (e.g., 5 degrees at 64x32 resolution)
5. **UV coordinates** — range normalized to [0, 1], monotonically increasing along grid
6. **Vertex stride** — all vertices packed as stride-8 (pos + normal + uv)
7. **Resolution warning** — log warning emitted when uSteps * vSteps > 1,000,000
8. **Degenerate input** — uSteps = 1 or vSteps = 1: produces valid (if minimal) mesh
9. **Triangle winding** — consistent winding order across all quads (no flipped triangles)
10. **Degenerate normal** — at sphere poles, normals should be valid (not NaN/zero)
11. **SceneClassifier integration** — `"parametric"` classified as `SceneType.TriangleMeshes`

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `menger-app/.../menger/objects/ParametricTessellator.scala` | **New** — tessellation logic |
| `menger-app/.../menger/dsl/SceneObject.scala` | Add `ParametricSurface` case class |
| `menger-common/.../menger/common/ObjectType.scala` | Add `"parametric"` to valid types |
| `menger-app/.../menger/engines/scene/MeshFactory.scala` | Add `"parametric"` case |
| `menger-app/.../menger/engines/scene/TriangleMeshSceneBuilder.scala` | Add `"parametric"` to private `isTriangleMeshType` |
| `menger-app/.../menger/ObjectSpec.scala` | Add optional `meshData: Option[TriangleMeshData]` field |
| `menger-app/.../menger/dsl/SceneConverter.scala` | Add `ParametricSurface` case to exhaustive match |
| `menger-app/src/test/scala/.../ParametricTessellatorSuite.scala` | **New** — unit tests |
| `menger-app/src/main/scala/examples/dsl/Parametric*.scala` | **New** — example scenes |
| `scripts/manual-test.sh` | Add interactive test entries |
| `ROADMAP.md` | Add analytical custom intersection to backlog |

---

## Backlog Addition

Add to ROADMAP.md long-term backlog:

| Idea | Description | Complexity |
|------|-------------|------------|
| Analytical ray intersection for known shapes | Custom OptiX intersection programs for cylinder, cone, torus (exact, no tessellation) | Medium |

Low priority — only pursue when a concrete need for better accuracy or performance is demonstrated.

---

## Out of Scope

- CLI `--objects type=parametric` support (lambdas can't be CLI strings)
- User-supplied normal functions (numerical derivatives sufficient)
- Adaptive tessellation / LOD (uniform grid is sufficient for now)
