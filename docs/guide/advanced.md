# Menger — Advanced Features

**Version**: 0.5.5
**Last Updated**: March 2026

← [Usage & Rendering](user-guide.md) | [User Guide Index](../USER_GUIDE.md)

---

## Advanced Features

### Animations

Generate frame sequences for animations using the `--animate` flag.

#### Animation Syntax

```bash
--animate frames=<N>:<param>=<start>-<end>[:<param2>=<start2>-<end2>...]
```

#### Animatable Parameters

**3D Rotation:**
- `rot-x`: Rotation around X axis (degrees)
- `rot-y`: Rotation around Y axis
- `rot-z`: Rotation around Z axis

**4D Rotation:**
- `rot-x-w`: Rotation around XW plane
- `rot-y-w`: Rotation around YW plane
- `rot-z-w`: Rotation around ZW plane

**4D Projection:**
- `projection-screen-w`: Projection screen distance
- `projection-eye-w`: Eye position in W dimension

**Fractal Level:**
- `level`: Recursion level (supports fractional values)

#### Animation Examples

**Simple Rotation:**
```bash
# Rotate 360° around Y axis over 36 frames
sbt "run --level 2 --save-name frame%03d.png \
    --animate frames=36:rot-y=0-360"
```

**Level Animation:**
```bash
# Animate from level 0 to level 3 over 30 frames
sbt "run --save-name level%03d.png \
    --animate frames=30:level=0-3"
```

**Combined Parameters:**
```bash
# Rotate and zoom simultaneously
sbt "run --save-name combined%03d.png \
    --animate frames=20:level=0-2:rot-y=0-90"
```

**Chained Animations:**
```bash
# First rotate in 4D, then rotate in 3D
sbt "run --sponge-type tesseract --save-name anim%03d.png \
    --animate frames=10:rot-x-w=0-45 \
    --animate frames=10:rot-y=0-90"
```

**Note:** Parameters specified in `--animate` cannot also be specified as CLI options (e.g., can't use both `--level 2` and `frames=10:level=0-2`).

#### Creating Videos from Frames

After generating frames, use ffmpeg to create a video:

```bash
# Generate frames
sbt "run --optix --objects 'type=sphere' --level 2 \
    --save-name frame%03d.png --animate frames=36:rot-y=0-360 \
    --timeout 0.5"

# Create MP4 video (30 FPS)
ffmpeg -framerate 30 -i frame%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Animated Scenes (t-Parameter)

**Introduced in v0.5.2**

The t-parameter animation system lets you create animated scenes by defining a `scene(t: Float)` function in the DSL. The renderer evaluates the function for each frame, performing a full scene rebuild per frame.

#### Defining an Animated Scene

Instead of `val scene: Scene`, define `def scene(t: Float): Scene`:

```scala
package examples.dsl

import scala.language.implicitConversions
import menger.dsl._

object OrbitingSphere:
  def scene(t: Float): Scene =
    val x = 2f * math.cos(t).toFloat
    val z = 2f * math.sin(t).toFloat
    Scene(
      camera = Camera(position = (0f, 3f, 6f), lookAt = (0f, 0f, 0f)),
      objects = List(
        Sphere(pos = Vec3(x, 0f, z), material = Material.Chrome, size = 0.5f)
      ),
      lights = List(
        Directional(direction = (1f, -1f, -1f), intensity = 2.0f)
      ),
      plane = Some(Plane(Y at -1.5, color = "#FFFFFF"))
    )
```

The `t` parameter is user-defined — it can represent an angle, a time value, a fractal level, or anything else. The scene function maps `t` to a complete scene description.

#### Freeze-Frame Rendering

Evaluate an animated scene at a single `t` value:

```bash
# Render OrbitingSphere at t=0.5
sbt "run --optix --scene examples.dsl.OrbitingSphere --t 0.5 --save-name orbit.png --headless"

# Render PulsingSponge at t=2.0 (fractal level 2)
sbt "run --optix --scene examples.dsl.PulsingSponge --t 2.0 --save-name pulse.png --headless"
```

Without `--t`, animated scenes default to `t=0`.

#### Multi-Frame Animation

Sweep `t` across a range to generate a frame sequence:

```bash
# 100-frame orbit animation, t from 0 to 2π
sbt "run --optix --scene examples.dsl.OrbitingSphere \
    --frames 100 --start-t 0 --end-t 6.28 \
    --save-name orbit_%04d.png --headless"

# Pulsing sponge from level 0 to 3
sbt "run --optix --scene examples.dsl.PulsingSponge \
    --frames 60 --start-t 0 --end-t 3 \
    --save-name pulse_%04d.png --headless"
```

The `t` value is linearly interpolated: `t = startT + frameIndex * (endT - startT) / (frames - 1)`.

#### CLI Options

```bash
--t <float>            Evaluate animated scene at fixed t (freeze-frame)
--start-t <float>      Start of t range (default: 0)
--end-t <float>        End of t range (default: 1)
--frames <int>         Number of frames to render (requires --save-name with %)
```

**Validation rules:**
- `--t` is mutually exclusive with `--start-t`, `--end-t`, `--frames`
- `--t` and `--frames` require `--scene` and `--optix`
- `--t` and `--frames` are mutually exclusive with `--animate` (the LibGDX animation system)
- `--frames` requires `--save-name` containing `%` for frame numbering

#### Creating Videos from Animated Scenes

```bash
# Generate frames
sbt "run --optix --scene examples.dsl.OrbitingSphere \
    --frames 120 --start-t 0 --end-t 6.28 \
    --save-name orbit_%04d.png --headless"

# Convert to MP4
ffmpeg -framerate 30 -i orbit_%04d.png -c:v libx264 -pix_fmt yuv420p orbit.mp4
```

#### Included Animated Examples

- **OrbitingSphere** — Chrome sphere orbiting the origin in the XZ plane. `t` maps to angle (radians); use `t` from 0 to 2π for a full orbit.
- **PulsingSponge** — Gold volume-filling sponge with varying fractal level. `t` controls the level (clamped to 0–3).

#### Backward Compatibility

Static scenes (`val scene: Scene`) continue to work unchanged. The `SceneLoader` auto-detects whether a scene object has a `def scene(Float)` method or a `val scene` field via reflection.

### Caustics (Light Focusing Effects)


Caustics are the patterns of light focused through transparent refractive objects (like the dancing light at the bottom of a pool). Menger implements caustics using **Progressive Photon Mapping (PPM)**.

#### Enabling Caustics

```bash
--caustics                        # Enable caustics rendering
--caustics-photons <int>          # Photons per iteration (default: 100000)
--caustics-iterations <int>       # PPM iterations (default: 10)
--caustics-radius <float>         # Initial gather radius (default: 0.1)
--caustics-alpha <float>          # Radius reduction factor (default: 0.7)
```

#### Caustics Examples

**Basic Caustics:**
```bash
# Glass sphere with caustics
sbt "run --optix --objects 'type=sphere:ior=1.5' --caustics"
```

**High-Quality Caustics:**
```bash
# More photons and iterations for better quality
sbt "run --optix --objects 'type=sphere:ior=1.5' \
    --caustics --caustics-photons 500000 --caustics-iterations 50"
```

**Caustics with Complex Geometry:**
```bash
# Menger sponge with caustics (computationally intensive!)
sbt "run --optix --objects 'type=sponge-surface:level=2:ior=1.5' \
    --caustics --caustics-photons 200000"
```

#### Performance Tips

- Start with default settings (100k photons, 10 iterations)
- Increase `--caustics-photons` for smoother caustics
- Increase `--caustics-iterations` for more accurate energy distribution
- Decrease `--caustics-radius` for finer detail (but may need more photons)
- Caustics significantly increase render time (10x-100x slower)

For more details, see [docs/caustics/CAUSTICS.md](../caustics/CAUSTICS.md).

#### Known Limitations

- **Non-deterministic results:** PPM uses stochastic photon tracing. Each render produces slightly
  different caustic patterns; pixel-exact reproduction is not guaranteed between runs.
- **General geometry supported:** Caustics work correctly for spheres, parametric surfaces
  (torus, Klein bottle, etc.), and any other refractive triangle mesh. Complex geometry
  (Menger sponges, tesseracts) may show weaker caustic patterns due to photon distribution.
- **Single light source:** Photons are emitted from the first light only; multi-light caustics
  are not yet supported.
- **Single plane deposition:** Caustic photons are deposited on the first ground plane only.
- **No interaction with colored shadows:** Caustics are computed in a separate PPM pass and do not
  interact with `--transparent-shadows` attenuation. Enabling both simultaneously is valid but
  the two effects are computed independently.
- **Interactive mode:** In interactive (non-headless) mode, caustics quality improves progressively —
  each rendered frame adds one PPM iteration. Full convergence requires `--caustics-iterations`
  frames to elapse.

### Antialiasing

Reduce jagged edges and improve quality with antialiasing.

#### LibGDX Mode MSAA

```bash
--antialias-samples <N>      # Number of MSAA samples (2, 4, 8, 16)
```

Example:
```bash
sbt "run --level 2 --antialias-samples 8"
```

#### OptiX Mode Recursive AA

OptiX mode uses recursive adaptive antialiasing that samples more heavily at edges:

```bash
--antialiasing                   # Enable adaptive AA
--aa-max-depth <int>             # Recursion depth (1-4, default: 2)
--aa-threshold <float>           # Edge detection threshold (0.0-1.0, default: 0.1)
```

**How it works:**
- Renders the scene at base resolution
- Detects edges by comparing adjacent pixel colors
- Recursively subdivides pixels that exceed the threshold
- More samples at edges, fewer in smooth areas

**Examples:**
```bash
# Standard quality
sbt "run --optix --objects 'type=sphere' --antialiasing"

# High quality (more recursion)
sbt "run --optix --objects 'type=sphere' \
    --antialiasing --aa-max-depth 4 --aa-threshold 0.05"

# Fast AA (less sensitive edge detection)
sbt "run --optix --objects 'type=sphere' \
    --antialiasing --aa-max-depth 2 --aa-threshold 0.2"
```

**Performance:**
- Depth 2: ~2-4x slower
- Depth 3: ~5-10x slower
- Depth 4: ~10-20x slower
- Lower threshold = more pixels subdivided = slower but better quality

### Multiple Objects

Render complex scenes with multiple objects (OptiX v0.4+):

```bash
--objects "type=<type>:param=value:param2=value2..."
```

#### Object Parameters

**Basic Parameters (all types):**
```bash
type=<type>                  # sphere, cube, sponge-volume, sponge-surface, cube-sponge, tesseract
pos=<x,y,z>                  # Position (default: 0,0,0)
size=<float>                 # Size/scale factor (default: 1.0)
color=#RRGGBB[AA]            # Color with optional alpha
ior=<float>                  # Index of refraction
material=<preset>            # Material preset (see section 6.1)
texture=<path>               # Texture file path (PNG/JPEG)
roughness=<float>            # Custom roughness (0.0-1.0)
metallic=<float>             # Custom metallic (0.0-1.0)
specular=<float>             # Custom specular (0.0-1.0)
emission=<float>             # Emission intensity (0.0-10.0, for glowing objects)
```

**Sponge-specific:**
```bash
level=<float>                # Fractal level (required for sponges)
```

**Tesseract 4D Projection (tesseract type only):**
```bash
eye-w=<float>                # 4D eye W-coordinate (default: 3.0)
screen-w=<float>             # 4D screen W-coordinate (default: 1.5)
rot-xw=<degrees>             # XW plane rotation (default: 15)
rot-yw=<degrees>             # YW plane rotation (default: 10)
rot-zw=<degrees>             # ZW plane rotation (default: 0)
```

**Tesseract Edge Rendering (tesseract type only):**
```bash
edge-radius=<float>          # Cylinder edge radius (default: 0.02)
edge-material=<preset>       # Edge material (chrome, plastic, film, etc.)
edge-color=#RRGGBB           # Edge color override
edge-emission=<float>        # Edge emission (0.0-10.0, for glowing edges)
```

#### Multi-Object Examples

**Three Spheres with Different Materials:**
```bash
sbt "run --optix \
    --objects 'type=sphere:pos=-2,0,0:material=glass' \
    --objects 'type=sphere:pos=0,0,0:material=gold' \
    --objects 'type=sphere:pos=2,0,0:material=chrome'"
```

**Mixed Geometry:**
```bash
sbt "run --optix \
    --objects 'type=sponge-surface:level=2:pos=0,0,0:material=diamond' \
    --objects 'type=sphere:pos=3,0,0:size=0.5:color=#FF0000' \
    --objects 'type=cube:pos=-3,0,0:material=copper'"
```

**Textured Objects:**
```bash
sbt "run --optix \
    --objects 'type=cube:texture=wood.png:pos=0,0,0' \
    --objects 'type=cube:texture=metal.jpg:pos=2,0,0'"
```

**Tesseract (4D Hypercube):**
```bash
# Basic tesseract with glass material
sbt "run --optix \
    --objects 'type=tesseract:material=glass'"

# Tesseract with custom 4D rotation
sbt "run --optix \
    --objects 'type=tesseract:material=diamond:rot-xw=30:rot-yw=45'"

# Tesseract with glowing edges (wireframe effect)
sbt "run --optix \
    --objects 'type=tesseract:material=glass:edge-radius=0.02:edge-material=chrome:edge-emission=3.0'"

# Tesseract with colored emissive edges
sbt "run --optix \
    --objects 'type=tesseract:edge-radius=0.025:edge-color=#00FFFF:edge-emission=5.0'"
```

> **DSL users:** The Scala DSL for Scene Description has its own reference document.
> See [DSL Reference](dsl-reference.md).

---

← [Usage & Rendering](user-guide.md) | [User Guide Index](../USER_GUIDE.md) | → [DSL Reference](dsl-reference.md)
