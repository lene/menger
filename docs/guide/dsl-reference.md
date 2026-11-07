# Menger — Scala DSL Reference

**Version**: 0.9.0
**Last Updated**: September 2026

← [Advanced Features](advanced.md) | [User Guide Index](../USER_GUIDE.md)

---

## Scala DSL for Scene Description

**Introduced in v0.5.0**

The Scala DSL provides a type-safe, IDE-friendly way to define scenes. Instead of specifying scenes via command-line arguments, you write Scala code that describes your scene structure. Scenes can live inside the project (compiled with Menger) or in a standalone `.scala` file that Menger compiles at startup — no rebuild needed.

### Why Use the DSL?

**Advantages over CLI:**
- **Type safety**: Catch errors at compile time, not runtime
- **IDE support**: Auto-completion, type hints, and refactoring
- **Reusability**: Import and share materials, lighting setups, and scene components
- **Version control**: Track scene changes in git alongside code
- **Composition**: Build complex scenes from reusable parts
- **Readability**: Cleaner syntax for complex scenes

**When to use CLI vs DSL:**
- CLI: Quick experiments, one-off renders, shell scripts
- DSL: Production scenes, complex setups, reusable components, team projects

### Basic DSL Structure

Create a Scala file in `menger-app/src/main/scala/examples/dsl/`:

```scala
package examples.dsl

import scala.language.implicitConversions
import menger.dsl.*

object MyScene:
  val scene = Scene(
    camera = Camera(
      position = (0f, 2f, 5f),
      lookAt = (0f, 0f, 0f)
    ),
    objects = List(
      Sphere(
        pos = (0f, 0f, 0f),
        material = Material.Chrome,
        size = 1.0f
      )
    ),
    lights = List(
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      )
    )
  )

  // Optional: Register a short name
  SceneRegistry.register("my-scene", scene)
```

**Load and render:**
```bash
# By fully-qualified class name
sbt "run --scene examples.dsl.MyScene"

# By registry short name (if registered)
sbt "run --scene my-scene"

# From a standalone file, compiled at startup (works with the packaged release too)
./menger-app-0.9.0/bin/menger-app --scene my_scene.scala
```

**Standalone scene files** must contain a top-level `object` (optionally inside a `package`)
with either `val scene: Scene` or `def scene(t: Float): Scene`, within the first 60 lines.
They are compiled against a *restricted* classpath (since 0.9.0): the Scala library,
`menger-common`, `scala-logging`, and menger's `menger.dsl`, `menger.objects` and `menger.video`
packages. Imports of LibGDX, `io.github.lene.optix`, `upickle`, or menger's `engines`, `tools`,
`cli` or `input` packages no longer compile. This is deliberate: a scene file is treated as
untrusted input and the compile step is the boundary (architecture decision SA-AD-4). `SceneRegistry.register` is not needed for file
scenes. A compile error is reported with the compiler's diagnostics and the run exits.

**Animation length:** an animated scene (`def scene(t: Float)`) can also declare
`val durationSeconds = <seconds>f`. `t` is then time in seconds, and `--scene` without
`--frames`/`--t`/`--save-name`/`--headless` opens a window that plays it in real time,
looping. Scenes without it behave as before. The older name `val duration` is still read,
with a deprecation warning.

**Live reload:** `--scene my_scene.scala` in an interactive window watches the file and
reloads it on every save — geometry, lights, planes, background and render settings update,
the camera and any 4D rotation stay where they are. A save that fails to compile is logged and
the window keeps its current scene. Only static scenes reload live; turning the file into an
animated scene needs a restart.

### Core DSL Types

**Vec3 - 3D Vectors:**
```scala
// Explicit construction
val pos = Vec3(1f, 2f, 3f)

// Tuple syntax (implicit conversion)
val pos = (1f, 2f, 3f)  // Converts to Vec3

// Also works with Int and Double
val pos = (1, 2, 3)      // Converts to Vec3(1f, 2f, 3f)
val pos = (1.0, 2.0, 3.0)  // Converts to Vec3(1f, 2f, 3f)
```

**Color - RGBA Colors:**
```scala
// Hex strings
val red = Color("#FF0000")
val transparentBlue = Color("#0000FF80")  // With alpha

// RGB/RGBA components (0.0-1.0)
val green = Color(0f, 1f, 0f)
val semiTransparent = Color(1f, 1f, 1f, 0.5f)

// Presets
Color.White, Color.Black, Color.Red, Color.Green, Color.Blue

// Implicit conversion from strings
val color: Color = "#FF0000"
```

An object's `color` together with a `material` tints only the material's RGB; transparency
stays the material's own (a `color` on `Material.Glass` keeps it transparent). Without a
`material`, the colour's alpha sets the object's opacity.

**Material - PBR Materials:**
```scala
// Presets
Material.Glass      // Transparent glass (IOR 1.5)
Material.Water      // Clear water (IOR 1.33)
Material.Diamond    // Crystal (IOR 2.42)
Material.Chrome     // Shiny metal
Material.Gold       // Metallic gold
Material.Copper     // Metallic copper

// Customize presets with .copy()
val brushedGold = Material.Gold.copy(roughness = 0.4f)
val tintedGlass = Material.Glass.copy(
  color = Color(0.9f, 0.95f, 1.0f, 0.02f)
)

// Factory methods
Material.matte(Color.Red)            // Matte red
Material.plastic(Color("#00FF00"))   // Green plastic
Material.metal(Color.White)          // Generic metal
Material.glass(Color.Blue)           // Blue glass

// Fully custom
Material(
  color = Color("#FF0000"),
  ior = 1.5f,
  roughness = 0.3f,
  metallic = 0.8f,
  specular = 0.9f
)
```

### Scene Objects

All scene objects share a common `rotation: Vec3` field for 3D rotation in degrees
(Sprint 19.7). Default is `Vec3.Zero` (no rotation):

```scala
// Rotate a sphere 45° around X and 30° around Y
Sphere(
  pos = (0f, 0f, 0f),
  material = Material.Chrome,
  size = 1.0f,
  rotation = Vec3(45f, 30f, 0f)
)

// Rotate a cube
Cube(
  material = Material.Gold,
  rotation = Vec3(0f, 60f, 0f)
)
```

The `rotation` field is available on every scene object.

**4D polytopes:** `Pentachoron` (5-cell), `Hexadecachoron` (16-cell), `Icositetrachoron`
(24-cell), `Hecatonicosachoron` (120-cell) and `Hexacosichoron` (600-cell) take the same
fields as `Tesseract` (`pos`, `size`, `material`, `color`, `projection`, `edgeRadius`,
`edgeMaterial`, `rotation`, ...). See the `PolytopeGallery` example.

**Platonic solids:** `Tetrahedron`, `Octahedron`, `Dodecahedron` and `Icosahedron` take the
same fields as `Cube`, edge tubes included (`edgeRadius`, `edgeMaterial`).

**Cone:** the fields of `Sphere`, plus optional `apex`, `base` (world coordinates) and
`radius`. Without them the cone is centred on `pos`, `size` high along y, with a base radius
of `size / 2`:

```scala
Cone(material = Some(Material.Copper), size = 1.5f)
Cone(apex = Some(Vec3(0f, 2f, 0f)), base = Some(Vec3(0f, 0f, 0f)), radius = Some(0.5f))
```

**Cylinder:** a capped cylinder with the fields of `Cone`, but optional `p0`, `p1` (the centres
of its end caps, world coordinates; both or neither) and `radius` instead of `apex` and `base`.
Without them the cylinder is centred on `pos`, `size` high along y (turned by `rotation`), with
a radius of `size / 2`. A radius that is not positive and finite, a `size` of 0 or less, only
one of `p0`/`p1`, non-finite end points and `p0 == p1` are rejected. It is shaded opaque,
diffuse or metallic: a transparent material (Glass, Film, any colour alpha below 1) renders
opaque, and one with an IOR above 1.05 (the refractive ones, and Plastic) receives no caustic;
the scene validator reports both (`opaque-only`). It has no UV coordinates, so image textures
and texture maps have no effect on it. Procedural colouring works, but the object-local one
(`proceduralType` 11, xyz_rgb_local) spans the box `pos` ± `size / 2`, not `p0`/`p1`: with
`p0`/`p1`, set `pos` and `size` so that box covers the cylinder. See the `Plinth` example:

```scala
Cylinder(material = Some(Material.Matte))               // centred on pos, 1 high, radius 0.5
Cylinder(p0 = Some(Vec3(0f, -0.65f, 0f)), p1 = Some(Vec3(0f, -0.45f, 0f)),
  radius = Some(1.8f), material = Some(Material.matte(Color(0.85f, 0.82f, 0.76f))))
```

**4D IFS fractals:** `Menger4D` (the `TesseractSponge`'s shape), `PentachoronGasket` and
`HexadecachoronGasket` (Sierpinski gaskets of the 5-cell and the 16-cell) take a `level` and
the fields of `Tesseract`. They are traced on the GPU without a stored mesh.
`PentachoronGasket` was called `Sierpinski4D`; the old name still compiles, deprecated.

Every command-line object type has a DSL class.

**Sphere:**
```scala
// Material only (at origin, default size)
Sphere(Material.Glass)

// Position and material
Sphere(
  pos = (-2f, 0f, 0f),
  material = Material.Chrome
)

// All parameters including rotation (Sprint 19.7)
Sphere(
  pos = (1f, 0f, 0f),
  material = Material.Gold,
  size = 1.5f,
  rotation = Vec3(0f, 45f, 0f)
)

// Color instead of material
Sphere(
  pos = (0f, 1f, 0f),
  color = Some(Color("#FF0000")),
  size = 0.8f
)
```

**Cube:**
```scala
// Same parameter patterns as Sphere
Cube(Material.Chrome)
Cube(pos = (2f, 0f, 0f), material = Material.Gold, size = 1.2f)
```

**Sponge:**
```scala
// Four sponge types
VolumeFilling      // Standard Menger sponge (20 cubes per iteration)
SurfaceUnfolding   // Surface-based sponge (12 faces per iteration)
CubeSponge         // Deprecated (menger#76): one instance per cube, levels up to 3; use VolumeFilling
RecursiveIAS       // Recursive IAS sponge — supports fractional levels (1–13)

// Basic sponge
Sponge(
  spongeType = VolumeFilling,
  level = 2f,
  material = Material.Glass
)

// With position and size
Sponge(
  pos = (-3f, 0f, 0f),
  spongeType = SurfaceUnfolding,
  level = 2.5f,
  material = Material.Chrome,
  size = 2.0f
)
```

**Tesseract (4D Hypercube):**
```scala
// Basic tesseract with default 4D projection
Tesseract(Material.Glass)

// With position and size
Tesseract(
  pos = (0f, 0f, 0f),
  material = Material.Chrome,
  size = 0.8f
)

// Custom 4D rotation (rotations in XW, YW, ZW planes)
Tesseract(Material.Glass).copy(
  projection = Some(Projection4DSpec(
    eyeW = 3.0f,      // 4D camera distance
    screenW = 1.5f,   // 4D projection plane
    rotXW = 30f,      // Rotation in XW plane (degrees)
    rotYW = 20f,      // Rotation in YW plane (degrees)
    rotZW = 10f       // Rotation in ZW plane (degrees)
  ))
)

// With edge rendering (wireframe overlay)
Tesseract(Material.Glass).copy(
  edgeRadius = Some(0.025f),
  edgeMaterial = Some(Material.Chrome)
)

// Color instead of material
Tesseract(
  pos = (1f, 0f, 0f),
  color = Some(Color("#4488FF")),
  size = 1.0f
)
```

**A 4D object in 5D to 8D (`ProjectionSpec`):**

`projection` takes a `ProjectionSpec` instead of the 4D shorthand `Projection4DSpec`. The
object (a `Tesseract`, `TesseractSponge` or a regular polytope) keeps its x y z w coordinates
and gets zeros in the further axes, is turned in the planes you list, and is projected down to
3D one dimension at a time. The axes are `x y z w v u t s` (5D adds `v`, 6D `u`, 7D `t`, 8D
`s`).

```scala
import menger.spec.{PlaneRotation, ProjectionSpec, ProjectionStage}

Tesseract(
  material = Some(Material.Film),
  size = 1.8f,
  projection = Some(ProjectionSpec(
    dimension = 5,                    // 4 to 8
    planes = List(
      PlaneRotation("xv", 30f),       // turns first (degrees)
      PlaneRotation("wv", 20f)        // then this one
    ),
    eye = 6f, screen = 1.2f           // for every stage ...
    // stages = List(ProjectionStage(6f, 1.2f), ProjectionStage(4f, 1f))  // ... or one per stage
  ))
)
```

- `planes` are applied in the order written: the first turns first, and the result depends on
  the order. A plane names two distinct axes that exist in `dimension`; for a positive angle
  the second axis turns towards the first (v towards x in `xv`, as w does for a positive
  `rotXW`). A plane of axes beyond `w` turns nothing until an
  earlier plane has moved the object into them.
- The projection has `dimension - 3` stages, the first dropping the highest axis. A stage scales
  a point by `(eye - screen) / (eye - c)`, `c` being the coordinate it drops, so the stages
  compound: with the default `eye` 3 and `screen` 1.5 a flat object is at 0.5 per stage. Give
  `stages` (one `ProjectionStage(eye, screen)` per stage) to set each pair; left empty, every
  stage uses `eye` and `screen`.
- It renders on the CPU through the projection chain, without the 4D fast paths: an animation
  rebuilds the object every frame, a sponge needs a whole `level`, and the IFS types
  (`Menger4D`, `PentachoronGasket`, `HexadecachoronGasket`) take a `Projection4DSpec` only.
  Shift+drag and Shift+scroll (eye) in the render window leave it alone (the log says so
  once), and the command line has no nD projection: `--objects` keeps the 4D keys.
- A `ProjectionSpec` has no `wScale`: the cube-to-tesseract extrusion (`CubeToTesseract`)
  cannot be combined with a lift into 5D or more.
- `ProjectionSpec(4, planes = List(PlaneRotation("zw", c), PlaneRotation("yw", b),
  PlaneRotation("xw", a)))` is `Projection4DSpec(rotXW = a, rotYW = b, rotZW = c)`.
- Example: `TesseractToPenteract`, a tesseract with a strong 5D perspective
  (`stages`), turned in `xv`, then `yv`, then `zv`.

**Hypercube (the cube of 4 to 8 dimensions):**

`Hypercube(dimension = N)` is the cube of N dimensions: 4 is the tesseract, 5 the penteract, 6
the hexeract, up to 8. It has 2^N vertices, N * 2^(N-1) edges and C(N,2) * 2^(N-2) square faces
(5: 32, 80, 80; 8: 256, 1024, 1792). `size` is the edge length, `pos` the centre. It has the
fields of `Tesseract`.

```scala
import menger.spec.{PlaneRotation, ProjectionSpec}

Hypercube(
  dimension = 6,
  material = Some(Material.Film),             // the faces
  size = 1.6f,
  edgeRadius = Some(0.012f),                  // one tube per edge: 192 for a 6-cube
  edgeMaterial = Some(Material.Gold),
  projection = Some(ProjectionSpec(
    dimension = 6,                            // N or more (a cube of N < D is lifted with zeros)
    planes = List(PlaneRotation("xu", 30f))   // a plane using the sixth axis u
  ))
)
```

- `projection` is a `ProjectionSpec` (see above) whose `dimension` is at least the cube's;
  left out, it is `ProjectionSpec(N)` with its defaults and no turn. A 6-cube with a
  `ProjectionSpec(5)` is rejected, the message naming both dimensions, as is a `dimension`
  outside 4 to 8.
- A cube of fewer dimensions than its view is lifted into it with zero coordinates, so a
  5-cube in `ProjectionSpec(7)` can be turned in planes using `u` and `t`: the plane `vt` moves
  it into `t`. A plane of axes the cube has not reached yet (`ts` for a 5-cube) turns nothing.
- A face or an edge tube with a vertex at or behind a stage's eye is left out, the rest is
  drawn. This holds for `Hypercube(4)` too, while the faces of a `Tesseract` are not clipped
  at an eye (lene/menger#176): the two differ there, edge tubes aside.
- It renders on the CPU through the chain, like any `ProjectionSpec` object: an animation
  rebuilds it every frame, and the window's Shift+drag leaves it alone. An 8-cube with edge
  tubes needs 1025 instances; the renderer's limit is raised for it unless you set
  `--max-instances` yourself.
- The command line has no `Hypercube`: `--objects` keeps the 4D `type=tesseract`.
- Example: `HexeractTurning`, a 6-cube with film faces and gold edges turning in `xu`.

**TesseractSponge (4D Menger Sponge):**
```scala
// Two sponge types
VolumeRemoving        // 4D volume-removing method (48 sub-tesseracts per level)
SurfaceSubdividing    // 4D surface subdivision method (16 faces per face)

// Basic tesseract sponge
TesseractSponge(
  spongeType = VolumeRemoving,
  level = 1f,
  material = Material.Glass
)

// With 4D rotation and projection
TesseractSponge(
  spongeType = SurfaceSubdividing,
  level = 1f,
  material = Material.Chrome
).copy(
  projection = Some(Projection4DSpec(
    eyeW = 4.0f,
    screenW = 2.0f,
    rotXW = 45f,
    rotYW = 30f
  ))
)

// Fractional level (smooth interpolation)
TesseractSponge(
  spongeType = VolumeRemoving,
  level = 1.5f,
  material = Material.Glass,
  size = 0.8f
)

// With edge rendering
TesseractSponge(
  pos = (0f, 0f, 0f),
  spongeType = VolumeRemoving,
  level = 1f,
  material = Material.Glass
).copy(
  edgeRadius = Some(0.015f),
  edgeMaterial = Some(Material.Film)
)
```

### Lights

**Directional Lights:** `direction` is the direction the light travels: `(0f, -1f, 0f)`
shines straight down, `(1f, -1f, -1f)` comes from above.
```scala
// Direction only (default intensity 1.0, white)
Directional(direction = (1f, -1f, -1f))

// With intensity
Directional(
  direction = (1f, -1f, -1f),
  intensity = 1.5f
)

// With color
Directional(
  direction = (0f, -1f, 0f),
  intensity = 2.0f,
  color = "#FFCC88"  // Warm white
)
```

**Point Lights:**
```scala
// Position only
Point(position = (0f, 5f, 0f))

// With intensity and color
Point(
  position = (3f, 4f, 3f),
  intensity = 2.0f,
  color = "#FF0000"  // Red
)
```

**Area Lights:**
```scala
// Disk light above the scene, facing down
AreaLight(
  position = (0f, 4f, 0f),
  normal = (0f, -1f, 0f),
  radius = 1.5f
)

// Brighter light with more shadow samples for softer penumbra
AreaLight(
  position = (2f, 5f, 1f),
  normal = (0f, -1f, 0f),
  radius = 2.0f,
  intensity = 1.5f,
  shadowSamples = 8
)
```

**Sun:** a distant disk-shaped key light given by the direction it travels and its apparent
diameter (menger#158). It becomes an `AreaLight` of that angular size, so its shadows have a
penumbra and it can emit caustic photons.
```scala
// The sun at its real size (0.53 degrees), 50 units away
Sun(direction = (-0.742f, -0.072f, -0.666f))

// Every parameter: Sun(direction, diameterDegrees, distance, color, intensity)
Sun(
  direction = (1f, -1f, -1f),
  diameterDegrees = 2f,        // above 0, at most 10; the sun is 0.53
  distance = 20f,              // only sets the penumbra width, see below
  color = Color(1f, 0.86f, 0.4f),
  intensity = 1.5f
)
```
- `direction` is the direction the light travels, as for `Directional`; it need not be
  normalized but must not be zero.
- `intensity` means what it means for `Directional`: `Sun(intensity = k)` lights the origin like
  `Directional(intensity = k)`, whatever the `distance`. Internally the area light's intensity is
  scaled by `1 + distance^2`, because area lights fall off as `1 / (1 + d^2)`. `distance` then
  only changes the penumbra width: the disk's radius is `distance * tan(diameterDegrees / 2)`,
  and a larger distance with the same angle is a sharper shadow edge.
- Rejected at scene load, naming the field: a zero direction, `diameterDegrees` of 0 or less or
  above 10, `distance` of 0 or less, a negative `intensity`.
- Only the first light of a scene emits caustic photons: put the `Sun` first in `lights` when the
  scene has caustics.
- The environment map emits no photons, so for caustics under an HDRI sky a `Sun` has to stand
  where the HDRI's sun is; `menger.tools.HdriSun` reads that off the file (see
  [Finding the sun in an HDRI](user-guide.md#finding-the-sun-in-an-hdri)). With `ibl`
  on, the sun's light is then counted twice, once as sky and once as the `Sun`; how to balance the
  two is an open experiment (menger#171).
- Example: `examples.dsl.SunKeyLight`.

### Parametric Surfaces

Define arbitrary 3D surfaces using a Scala function `f(u, v) => Vec3`. The surface is
CPU-tessellated into a triangle mesh and rendered through the full OptiX pipeline.

```scala
// Parametric sphere (closed in U, open in V — seam-welded at poles)
ParametricSurface(
  f = (u, v) => Vec3(
    math.cos(u).toFloat * math.sin(v).toFloat,
    math.cos(v).toFloat,
    math.sin(u).toFloat * math.sin(v).toFloat
  ),
  uRange = (0f, 2f * math.Pi.toFloat),
  vRange = (0f, math.Pi.toFloat),
  closedU = true,
  material = Some(Material.Glass)
)

// Torus (closed in both U and V)
val R = 1.0f; val r = 0.35f
ParametricSurface(
  f = (u, v) => Vec3(
    (R + r * math.cos(v).toFloat) * math.cos(u).toFloat,
    r * math.sin(v).toFloat,
    (R + r * math.cos(v).toFloat) * math.sin(u).toFloat
  ),
  uRange = (0f, 2f * math.Pi.toFloat),
  vRange = (0f, 2f * math.Pi.toFloat),
  closedU = true, closedV = true,
  material = Some(Material.Glass)
)

// Wavy sheet (open surface, higher resolution)
ParametricSurface(
  f = (u, v) => Vec3(u, 0.3f * math.sin(u * 2).toFloat * math.cos(v * 2).toFloat, v),
  uRange = (-2f, 2f),
  vRange = (-2f, 2f),
  uSteps = 64, vSteps = 64,
  ior = 1.5f
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `f` | required | Surface function `(u, v) => Vec3` |
| `uRange` | `(0, 2π)` | Range of u parameter |
| `vRange` | `(0, π)` | Range of v parameter |
| `uSteps` | `64` | Tessellation resolution along u |
| `vSteps` | `32` | Tessellation resolution along v |
| `closedU` | `false` | Weld the seam at u boundaries (e.g. cylinder, torus) |
| `closedV` | `false` | Weld the seam at v boundaries |
| `material` | `None` | Material preset (Glass, Chrome, etc.) |
| `color` | `None` | RGBA color |
| `ior` | `1.0` | Index of refraction |
| `size` | `1.0` | Uniform scale |
| `pos` | `(0,0,0)` | Position offset |

**Built-in example scenes** (load with `--scene <name>`): `parametric-sphere`,
`parametric-torus`, `parametric-wavy-sheet`, `parametric-moebius`, `parametric-klein-bottle`,
plus the caustics variants `parametric-sphere-caustics`, `parametric-torus-caustics` and the
thin-film `parametric-klein-bottle-film`. See [All built-in scenes](#all-built-in-scenes).

### Camera and Plane

**Camera:**
```scala
// Default camera
Camera.Default  // Position (0, 0, 3), looking at origin

// Custom position and lookAt
Camera(
  position = (4f, 3f, 6f),
  lookAt = (0f, 0f, 0f)
)

// With custom up vector
Camera(
  position = (0f, 5f, 0f),
  lookAt = (0f, 0f, 0f),
  up = (0f, 0f, 1f)
)
```

**Plane (Ground/Floor):**
```scala
// Axis syntax: Y at -2 means Y-axis plane at position -2
Plane(Y at -2, color = "#808080")

// Checkered pattern
Plane.checkered(Y at -1, ("#FFFFFF", "#000000"))

// X or Z planes
Plane(X at 5, color = "#FF0000")   // Vertical wall
Plane(Z at -3, color = "#00FF00")  // Back wall

// Image texture and PBR maps (Task 21.6) — planar UV, repeating per world unit
Plane(Y at -1, texture = Some("stone.png"), normalMap = Some("stone_n.png"))
Plane(Y at -1, texture = Some("tile.png"), roughnessMap = Some("tile_r.png"))

// Rectangular video texture (resolved relative to --texture-dir)
Cube(videoTexture = Some(VideoTexture("video/two-frame-rgba.mov")))
```

### Caustics

```scala
// High-quality preset
Some(Caustics.HighQuality)  // 500K photons, 20 iterations

// Default settings
Some(Caustics.Default)       // 100K photons, 10 iterations

// Custom configuration
Some(Caustics(
  photons = 250000,
  iterations = 15,
  radius = 0.08f,
  alpha = 0.75f
))

// Disabled
None
```

### Fog

```scala
// Light blue-gray haze (default color)
fog = Some(Fog(density = 0.05f))

// Custom color and density
fog = Some(Fog(density = 0.15f, color = Color(0.9f, 0.9f, 0.95f)))

// Disabled (default)
fog = None
```

`density` controls attenuation rate: `exp(-density × distance)`.
At density=0.05 and distance=20, fog is ~37% of scene color.

### Environment Map

An equirectangular `.hdr` file rendered as the scene background. On its own it only paints the
background and lights nothing (the next sections' `ibl` makes it light the scene; it emits no
caustic photons either way, see `Sun`):

```scala
// Enable HDR background
envMap = Some("cliffside_2k.hdr")   // resolved relative to --texture-dir

// Disabled (default — solid background color)
envMap = None
```

The path is resolved relative to `--texture-dir` at render time. Only `.hdr` (Radiance RGBE)
is supported.

### Environment Map Video

An equirectangular 2:1 video can be used as the animated background. It uses the same
`VideoPlayback` controls as object video textures:

```scala
envMapVideo = Some(EnvMapVideo(
  "video/two-frame-equirect-rgba.mov",
  VideoPlayback(
    timeMapping = VideoTimeMapping.TProgress,
    repeat = VideoRepeat.Loop,
    fpsOverride = Some(2.0)
  )
))
```

`envMap` and `envMapVideo` are mutually exclusive. When `ibl` is enabled, the same
sampled video frame lights the scene.

### Tone Mapping

Controls how HDR values (> 1.0) are mapped to display range [0,1]:

```scala
// No tone mapping — clamp (default, fine for non-HDR scenes)
toneMapping = ToneMapping.None

// Reinhard — smooth roll-off, recommended with HDR backgrounds
toneMapping = ToneMapping.Reinhard(exposure = 1.2f)

// ACES Narkowicz 2015 filmic — stronger contrast S-curve
toneMapping = ToneMapping.ACES(exposure = 0.9f)
```

`exposure` is a pre-tone-map multiplier. Values above 1.0 brighten; below 1.0 darken.
Tone mapping applies only to the env map sample; scene geometry colors pass through unchanged.

### Complete Scene Example

```scala
package examples.dsl

import scala.language.implicitConversions
import menger.dsl.*

object GlassAndMetalScene:
  val scene = Scene(
    camera = Camera(
      position = (5f, 3f, 7f),
      lookAt = (0f, 0f, 0f)
    ),

    objects = List(
      // Left: Tinted glass sphere
      Sphere(
        pos = (-2.5f, 0f, 0f),
        material = Material.Glass.copy(
          color = Color(0.9f, 0.95f, 1.0f, 0.02f)
        ),
        size = 1.0f
      ),

      // Center: Chrome cube
      Cube(
        pos = (0f, 0f, 0f),
        material = Material.Chrome,
        size = 1.2f
      ),

      // Right: Brushed gold sphere
      Sphere(
        pos = (2.5f, 0f, 0f),
        material = Material.Gold.copy(roughness = 0.3f),
        size = 1.0f
      )
    ),

    lights = List(
      // Key light from upper right
      Directional(
        direction = (1f, -1f, -1f),
        intensity = 1.5f
      ),
      // Fill light from left
      Directional(
        direction = (-1f, -0.5f, -1f),
        intensity = 0.5f
      ),
      // Warm accent from above
      Point(
        position = (0f, 5f, 0f),
        intensity = 1.2f,
        color = "#FFCC88"
      )
    ),

    plane = Some(Plane(Y at -2, color = "#404040")),
    caustics = Some(Caustics.HighQuality)
  )

  SceneRegistry.register("glass-and-metal", scene)
```

### Reusable Components Pattern

**Create a materials library (`examples/dsl/common/Materials.scala`):**
```scala
package examples.dsl.common

import menger.dsl.*

object Materials:
  val TintedGlass = Material.Glass.copy(
    color = Color(0.85f, 0.9f, 1.0f, 0.02f)
  )

  val BrushedGold = Material.Gold.copy(roughness = 0.4f)

  val RoseGold = Material.metal(Color(0.85f, 0.5f, 0.5f)).copy(
    roughness = 0.1f
  )
```

**Create a lighting library (`examples/dsl/common/Lighting.scala`):**
```scala
package examples.dsl.common

import menger.dsl.*

object Lighting:
  val ThreePointLighting = List(
    Directional(direction = (1f, -1f, -1f), intensity = 1.5f),
    Directional(direction = (-1f, -0.5f, -1f), intensity = 0.5f),
    Directional(direction = (0f, 0.5f, 1f), intensity = 0.8f)
  )

  val GoldenHourLighting = List(
    Directional(
      direction = (1f, -0.3f, -1f),
      intensity = 1.8f,
      color = "#FFCC88"  // Warm orange
    ),
    Directional(
      direction = (-1f, 0.5f, 1f),
      intensity = 0.4f,
      color = "#88AAFF"  // Cool blue
    )
  )
```

**Use in your scene:**
```scala
package examples.dsl

import scala.language.implicitConversions
import menger.dsl.*
import examples.dsl.common.Materials.*
import examples.dsl.common.Lighting.*

object MyScene:
  val scene = Scene(
    camera = Camera.Default,
    objects = List(
      Sphere(pos = (-1f, 0f, 0f), material = TintedGlass),
      Sphere(pos = (1f, 0f, 0f), material = BrushedGold)
    ),
    lights = ThreePointLighting,
    plane = Some(Plane(Y at -1, color = "#606060"))
  )
```

### Included Example Scenes

All example scenes are in `menger-app/src/main/scala/examples/dsl/`. Load them by class name
(`--scene examples.dsl.ThreeMaterials`) or, where registered, by short name.

- **SimpleScene** - Minimal single chrome sphere
- **ThreeMaterials** - Glass, Chrome, Gold showcase with two lights
- **GlassSphere** - Glass sphere with caustics on white floor
- **CausticsDemo** - High-quality caustics demonstration
- **CustomMaterials** - Five custom materials using `.copy()` and factories
- **ComplexLighting** - Five-light setup (key, fill, rim, warm, cool)
- **SpongeShowcase** - Three sponge types comparison
- **MengerShowcase** - Classic Menger Sponge with three-point lighting
- **TesseractDemo** - 4D hypercube (tesseract) with custom projection and glass material
- **PolytopeGallery** - The five regular 4D polytopes besides the tesseract, side by side
- **FilmSphere** - Thin-film interference demonstration with Film material
- **ReusableComponents** - Demonstrates importing common materials/lighting
- **MixedMetallicShowcase** - Five spheres at metallic 0.0→1.0, same roughness
- **TwoSpheres**, **CausticsCanonical**, **CausticsReference** - Caustics validation scenes
- **EnvMapDemo**, **IblSphereDemo**, **DenoiseIblDemo** - HDR environment lighting
- **PrismDispersion**, **DiamondFire** - Wavelength-dependent refraction
- **TrefoilKnot** - B-spline curve primitive
- **VideoTextureCube** - Video texture on a cube
- **RenderSettingsDemo** - Render-quality settings from the DSL
- Animated (`def scene(t: Float)`, use with `--t`, `--frames` or `--preview`):
  **OrbitingSphere**, **PulsingSponge**, **SpongeLevelAnimation**, **RotatingSilverSponge**,
  **SierpinskiHDRRotation**, **FractalWithHDR** (HDR lighting), **EnvMapVideoSponge**
  (video environment map)

#### All Built-in Scenes

Short names accepted by `--scene` (registered in `SceneIndex`):

| Topic | Short names |
|-------|-------------|
| Basics & materials | `simple`, `three-materials`, `custom-materials`, `mixed-metallic`, `film-sphere`, `reusable-components`, `render-settings-demo` |
| Sponges & 4D | `menger-showcase`, `sponge-showcase`, `tesseract-demo`, `polytope-gallery` |
| Lighting & environment | `complex-lighting`, `env-map-demo`, `ibl-sphere-demo`, `denoise-ibl-demo` |
| Caustics | `glass-sphere`, `caustics-demo`, `two-spheres`, `two-spheres-off`, `caustics-canonical`, `caustics-canonical-off`, `caustics-canonical-parametric`, `caustics-canonical-parametric-off`, `caustics-reference`, `caustics-reference-default` |
| Parametric surfaces | `parametric-sphere`, `parametric-sphere-caustics`, `parametric-torus`, `parametric-torus-caustics`, `parametric-wavy-sheet`, `parametric-moebius`, `parametric-klein-bottle`, `parametric-klein-bottle-film` |
| Curves & textures | `trefoil-knot`, `video-texture-cube` |

The `-off` variants render the same scene with caustics disabled (for A/B comparison).

**Render an example:**
```bash
sbt "run --scene examples.dsl.ThreeMaterials"
sbt "run --scene glass-sphere"
sbt "run --scene menger-showcase"
```

### Tips and Best Practices

1. **Always import `scala.language.implicitConversions`** at the top of your scene files for tuple syntax to work
2. **Use `Some()` for optional fields** like `caustics` and `plane`
3. **Register short names** with `SceneRegistry.register()` for convenience
4. **Build component libraries** - create reusable materials and lighting in `common/` packages
5. **Use `.copy()`** to customize presets rather than defining from scratch
6. **Type safety** - let the compiler catch errors before runtime
7. **IDE support** - use IntelliJ IDEA or Metals for auto-completion and type hints

---

← [Advanced Features](advanced.md) | [User Guide Index](../USER_GUIDE.md) | → [Tutorials](tutorials.md)
