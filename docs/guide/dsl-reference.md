# Menger — Scala DSL Reference

**Version**: 0.5.8
**Last Updated**: May 2026

← [Advanced Features](advanced.md) | [User Guide Index](../USER_GUIDE.md)

---

## Scala DSL for Scene Description

**Introduced in v0.5.0**

The Scala DSL provides a type-safe, IDE-friendly way to define scenes that compile with your project. Instead of specifying scenes via command-line arguments, you write Scala code that describes your scene structure.

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
sbt "run --optix --scene examples.dsl.MyScene"

# By registry short name (if registered)
sbt "run --optix --scene my-scene"
```

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

The `rotation` field is available on: `Sphere`, `Cube`, `Sponge`, `Tesseract`,
`TesseractSponge`, `ParametricSurface`.

> **CLI-only types:** Platonic solids (tetrahedron, octahedron, dodecahedron, icosahedron),
> cone, and 4D polytopes (pentachoron, 16-cell, 24-cell, 120-cell, 600-cell) do not yet have
> Scala DSL constructors. Use the CLI `--objects type=<name>:...` syntax for those types.
> See [reference.md](reference.md) for the full parameter list.

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
// Three sponge types
VolumeFilling      // Standard Menger sponge (20 cubes per iteration)
SurfaceUnfolding   // Surface-based sponge (12 faces per iteration)
CubeSponge         // Cube-based sponge pattern

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

**Directional Lights:**
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

**Built-in example scenes** (load with `--dsl <name>`): `parametric-sphere`,
`parametric-torus`, `parametric-wavy-sheet`, `parametric-moebius`, `parametric-klein-bottle`.

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

All example scenes are in `menger-app/src/main/scala/examples/dsl/`:

- **SimpleScene** - Minimal single chrome sphere
- **ThreeMaterials** - Glass, Chrome, Gold showcase with two lights
- **GlassSphere** - Glass sphere with caustics on white floor
- **CausticsDemo** - High-quality caustics demonstration
- **CustomMaterials** - Five custom materials using `.copy()` and factories
- **ComplexLighting** - Five-light setup (key, fill, rim, warm, cool)
- **SpongeShowcase** - Three sponge types comparison
- **MengerShowcase** - Classic Menger Sponge with three-point lighting
- **TesseractDemo** - 4D hypercube (tesseract) with custom projection and glass material
- **FilmSphere** - Thin-film interference demonstration with Film material
- **ReusableComponents** - Demonstrates importing common materials/lighting
- **MixedMetallicShowcase** - Five spheres at metallic 0.0→1.0, same roughness

**Render an example:**
```bash
sbt "run --optix --scene examples.dsl.ThreeMaterials"
sbt "run --optix --scene glass-sphere"
sbt "run --optix --scene menger-showcase"
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
