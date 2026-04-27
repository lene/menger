# Menger — Usage & Rendering

**Version**: 0.5.7
**Last Updated**: March 2026

← [Quick Start](quickstart.md) | [User Guide Index](../USER_GUIDE.md)

---

## Basic Usage

### Running the Application

The application is run through sbt with various command-line options:

```bash
# Interactive mode (LibGDX real-time preview)
sbt run

# With options (must quote the entire command)
sbt "run --option1 value1 --option2 value2"

# Examples
sbt "run --level 2"                          # Level 2 sponge
sbt "run --width 1920 --height 1080"        # HD resolution
sbt "run --optix --objects 'type=sphere'    # Ray-traced sphere
```

**Important**: Always quote the entire command after `run` when using options.

### Command-Line Options

#### General Options

```bash
--timeout <seconds>          # Auto-exit after specified time (useful for testing)
--width <pixels>             # Window width (default: 800)
--height <pixels>            # Window height (default: 600)
--save-name <pattern>        # Save frames (e.g., "frame%d.png")
```

#### LibGDX Mode Options

```bash
--sponge-type <type>         # Geometry type (see section 5)
--level <float>              # Recursion level (supports fractional, e.g., 1.5)
--lines                      # Render as wireframe
--color <rrggbb[aa]>         # Hex color (e.g., ff0000 for red)
--face-color <rrggbb[aa]>    # Face color with alpha (overlay mode)
--line-color <rrggbb[aa]>    # Line color with alpha (overlay mode)
--antialias-samples <int>    # MSAA samples for antialiasing
```

#### 4D Projection Options

```bash
--projection-screen-w <float>    # 4D projection screen distance
--projection-eye-w <float>       # 4D projection eye distance
--rot-x-w <float>                # 4D rotation around XW plane
--rot-y-w <float>                # 4D rotation around YW plane
--rot-z-w <float>                # 4D rotation around ZW plane
```

#### OptiX Mode Options

```bash
--optix                      # Enable OptiX ray tracing
--object <type>              # DEPRECATED: Use --objects instead
--radius <float>             # Object radius (default: 1.0)
--scale <float>              # Object scale factor (default: 1.0)
--center <x,y,z>             # Object center position
--objects type=sphere:ior=1.5  # Index of refraction via --objects (1.0=opaque, 1.5=glass, 2.42=diamond)

# Camera
--camera-pos <x,y,z>         # Camera position (default: 0,0.5,3)
--camera-lookat <x,y,z>      # Camera look-at target (default: 0,0,0)
--camera-up <x,y,z>          # Camera up vector (default: 0,1,0)

# Lighting
--light <spec>               # Add light (repeatable, max 8)
                             # Types: directional:x,y,z[:i[:c]], point:x,y,z[:i[:c]],
                             #        area:px,py,pz:nx,ny,nz:radius[:samples[:i[:c]]]
--shadows                    # Enable shadow rays

# Quality
--antialiasing               # Enable recursive adaptive antialiasing
--aa-max-depth <int>         # AA recursion depth (1-4, default: 2)
--aa-threshold <float>       # AA edge threshold (0.0-1.0, default: 0.1)
--max-ray-depth <int>        # Bounce / refraction recursion depth (1..8, default: 5)
--allow-uniform-render       # Disable the failed-render diagnostic (see "Render health checks")
--stats                      # Display ray tracing statistics

# Scene
--plane <spec>               # Ground plane (default: +y:-2)
--plane-color <spec>         # Plane color (solid: #RRGGBB, checkered: RRGGBB:RRGGBB)
--plane-material <name>      # Plane material preset (chrome, gold, glass, …)
--transparent-shadows        # Colored shadow tinting through transparent objects
```

### Interactive Controls

#### LibGDX Mode (Real-time Preview)

When running in LibGDX interactive mode, use these controls:

**Mouse Controls:**
- **Left Click + Drag**: Rotate the object around X and Y axes
- **Right Click + Drag**: Rotate the object around the Z axis
- **Scroll Wheel**: Zoom in/out

**Keyboard Shortcuts:**
- **ESC**: Exit application
- **Space**: Pause/resume animation (if enabled)
- **R**: Reset camera to default position
- **S**: Take screenshot (saves to current directory)
- **W**: Toggle wireframe mode
- **F**: Toggle fullscreen

**4D Rotation (Tesseract/4D objects):**
- **Shift + LEFT/RIGHT arrows**: Rotate in XW plane
- **Shift + UP/DOWN arrows**: Rotate in YW plane
- **Shift + PAGE_UP/PAGE_DOWN**: Rotate in ZW plane

#### OptiX Mode (Ray Tracing)

When running in OptiX mode, use these controls:

**Mouse Controls:**
- **Left Click + Drag**: Rotate camera view around the scene
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out

**Keyboard Shortcuts:**
- **ESC**: Reset 4D view to initial state (rotation and projection)
- **Ctrl + Q**: Exit application

**4D Rotation (Tesseract/4D objects):**

*Keyboard Controls:*
- **Shift + LEFT/RIGHT arrows**: Rotate tesseract in XW plane
- **Shift + UP/DOWN arrows**: Rotate tesseract in YW plane
- **Shift + PAGE_UP/PAGE_DOWN**: Rotate tesseract in ZW plane

*Mouse Controls:*
- **Shift + Left Drag (horizontal)**: Rotate tesseract in XW plane
- **Shift + Left Drag (vertical)**: Rotate tesseract in YW plane
- **Shift + Right Drag (vertical)**: Rotate tesseract in ZW plane

**4D Projection Adjustment:**
- **Shift + Scroll Wheel**: Adjust the 4D eye distance (`eyeW`). Scroll up = move the 4D viewpoint further away (flatter projection); scroll down = move closer (more perspective distortion).

**Note**: In OptiX mode, 4D rotation and projection changes trigger a scene rebuild and re-render. The tesseract will visibly change shape as it rotates through 4D space.

---

## Rendering Modes

### LibGDX Mode (Real-time Preview)

LibGDX mode uses OpenGL rasterization for real-time interactive rendering. This is the default mode when you run `sbt run` without `--optix`.

**Advantages:**
- Real-time interaction (60+ FPS)
- Works on any system with OpenGL support
- Immediate visual feedback
- Interactive camera control

**Limitations:**
- No refraction or realistic light transport
- Limited material support
- No caustics or advanced effects

**Best for:**
- Exploring geometry and composition
- Quick previews before final rendering
- Interactive demonstrations
- Prototyping animations

**Example:**
```bash
# Interactive level 3 Menger sponge
sbt "run --sponge-type square-sponge --level 3"

# Wireframe overlay on transparent faces
sbt "run --level 2 --face-color ffffff40 --line-color 000000ff"
```

### OptiX Mode (High-Quality Ray Tracing)

OptiX mode uses NVIDIA's ray tracing engine for physically-based rendering with realistic light transport.

**Advantages:**
- Physically accurate refraction (glass, water, diamond)
- Realistic shadows
- Caustics (light focusing through transparent objects)
- Recursive reflections
- Advanced material properties (metallic, roughness, IOR)
- High-quality antialiasing

**Limitations:**
- Requires NVIDIA GPU
- Slower rendering (seconds to minutes per frame)
- No real-time interaction during render
- Linux only

**Best for:**
- Final high-quality output
- Publication-quality images
- Realistic material showcase
- Complex lighting scenarios

**Example:**
```bash
# Glass sphere with refraction
sbt "run --optix --objects 'type=sphere:ior=1.5:size=1.5'"

# High-quality Menger sponge with shadows and antialiasing
sbt "run --optix --objects 'type=sponge-surface:level=2' \
    --shadows --antialiasing --plane-color ffffff:808080"
```

#### Recursion Depth (`--max-ray-depth`, Sprint 18.5)

`--max-ray-depth N` controls how many bounce / refraction recursions a single
camera ray may take inside the OptiX pipeline. Default is 5; the supported
range is 1..8. Increase the value when rendering deep glass stacks or chains
of internal reflections that would otherwise terminate as black:

```bash
# Five stacked glass spheres — needs depth ≥ 6 to see through all of them
menger --objects 'type=sphere:ior=1.5:x=-1' \
       --objects 'type=sphere:ior=1.5:x=-0.5' \
       --objects 'type=sphere:ior=1.5:x=0' \
       --objects 'type=sphere:ior=1.5:x=0.5' \
       --objects 'type=sphere:ior=1.5:x=1' \
       --max-ray-depth 8
```

Lower values render faster but truncate refractive light paths sooner.
Existing reference images are unchanged at the default; the flag only opts in
to a deeper traversal.

#### Render Health Checks (Sprint 18.6)

When a frame is saved, Menger inspects the pixel buffer and refuses to write
a PNG that consists entirely of one colour (≥ 99 % of pixels within ε of a
single RGB value). This catches the "all-red error fill / all-black no-trace
/ all-blue clear-colour" failure modes that otherwise produce silently broken
reference images. The CLI exits with status 2 and logs:

```
Failed render: all pixels are approximately (R,G,B); CLI args: …
```

Pass `--allow-uniform-render` if a uniform output is intentional (e.g. a
clear-colour smoke test). For the broader debugging method, use the
`debugging-rendering-bugs` skill.

---

## Geometry Types

### 3D Objects

#### Basic Primitives

**Sphere** (`--objects 'type=sphere'`)
- Perfect sphere primitive
- Supports all material properties
- Fast rendering
- Example: `sbt "run --optix --objects 'type=sphere:size=1.5'"`

**Cube** (`--objects 'type=cube'`)
- Triangle mesh cube
- Supports textures and all materials
- Example: `sbt "run --optix --objects 'type=cube:size=0.5'"`

#### Menger Sponges

**Surface Subdivision** (`--sponge-type square-sponge` or `--objects 'type=sponge-surface'`)
- Generates only the outer surface
- Computational complexity: O(12^n)
- Higher detail, suitable for levels 0-6
- No internal faces (efficient for ray tracing)
- Example: `sbt "run --sponge-type square-sponge --level 3"`

**Volume Subdivision** (`--sponge-type cube-sponge` or `--objects 'type=sponge-volume'`)
- Generates cube instances
- Computational complexity: O(20^n)
- Uses Instance Acceleration Structure (IAS)
- Efficient for high levels (5+) in OptiX mode
- Example: `sbt "run --optix --objects 'type=sponge-volume:level=5'"`

**Recursive IAS** (`--objects 'type=sponge-recursive-ias'`, OptiX only)
- One unit-cube GAS reused at every level via N nested instance acceleration
  structures driven by the 20 Menger sub-cube transforms
- Memory cost is O(level × 20) instead of O(20^level), so deep sponges that
  would otherwise exceed VRAM remain practical
- Integer levels 1..14 (capped by OptiX's MAX_TRAVERSABLE_GRAPH_DEPTH=16)
- Example: `sbt "run --optix --objects 'type=sponge-recursive-ias:level=6'"`

**Which to use?**
- **Surface subdivision**: Better for low to medium levels (0-4), more geometric detail, works in both modes
- **Volume subdivision**: Better for high levels (5+) in OptiX mode, faster with IAS optimization
- **Recursive IAS**: Use for very deep levels (6+) where the explicit volume mesh
  no longer fits in VRAM; renders the same shape with constant per-level memory

### 4D Objects

#### Tesseract (4D Hypercube)

**Tesseract** (`--sponge-type tesseract`)
- 4D analog of a cube
- 8 cubic "cells" (faces)
- Projected to 3D using 4D→3D perspective projection
- Example: `sbt "run --sponge-type tesseract"`

**Tesseract Sponge Volume** (`--sponge-type tesseract-sponge-volume`)
- 4D Menger sponge (48 tesseracts per iteration)
- Hausdorff dimension ≈ 3.524
- Example: `sbt "run --sponge-type tesseract-sponge-volume --level 2"`

**Tesseract Sponge Surface** (`--sponge-type tesseract-sponge-surface`)
- Alternative generation (16 faces per face)
- More efficient: O(16^n) vs O(48^n)
- Example: `sbt "run --sponge-type tesseract-sponge-surface --level 2"`

#### 4D Projection Controls

Control the 4D→3D projection:

```bash
# Adjust projection parameters
--projection-screen-w 1.0    # Distance to projection screen in W dimension
--projection-eye-w 3.0       # Eye position in W dimension

# Rotate in 4D space (individual axes)
--rot-x-w 45                 # Rotate around XW plane (degrees)
--rot-y-w 30                 # Rotate around YW plane
--rot-z-w 15                 # Rotate around ZW plane

# Compact shorthand for all three 4D rotation angles at once
--rotation-4d 45,30,15       # XW,YW,ZW in degrees (mutually exclusive with --rot-x-w/y-w/z-w)
```

#### GPU 4D Projection (Sprint 18.3)

By default the 4D rotation + perspective projection runs on the CPU at
scene-build time. For larger 4D meshes (especially `tesseract-sponge`
at level ≥ 2) and for animated 4D viewpoints this is the dominant cost.

```bash
--gpu-project-4d             # opt-in: run rotation+projection on the GPU
```

When the flag is set:

- **Setup time** — `tesseract-sponge level=2` drops from ≈4s on the CPU
  to ≈130ms (~30× faster); larger levels scale better.
- **Animation** — when an animation only changes 4D rotation
  (`--animate rot-x-w/y-w/z-w`) or projection eye/screen depth
  (`projection-eye-w`, `projection-screen-w`), the engine refits the
  existing GPU mesh in place via `updateMesh4DProjection` instead of
  rebuilding the scene. A 10-frame XW-rotation animation on
  tesseract-sponge level=2 is ≈300× faster (5.5ms vs 1500ms).
  Animations that change other parameters (size, material, position,
  level…) fall back to the rebuild path automatically.

The flag is purely opt-in: with the default off, behaviour and image
output are unchanged. Output of the GPU path matches the CPU path to
L∞ ≤ 6/255 (typically L∞ = 0). For arbitrary 4D meshes that are not
quad-based, decompose each polygon into a fan of degenerate quads
`(a, b, c, c)` — the kernel handles the degenerate-normal fallback
identically to the CPU path.

### Fractional Levels

All sponge types support fractional recursion levels (e.g., `--level 1.5`), which creates a smooth transition between integer levels using alpha blending.

**How it works:**
- The floor level (e.g., 1) is rendered with increasing transparency
- The ceiling level (e.g., 2) is rendered increasingly opaque
- Alpha varies linearly from 0.0 (floor level) to 1.0 (ceiling level)

**Examples:**
```bash
# Halfway between level 1 and level 2
sbt "run --level 1.5"

# Quarter of the way from level 2 to level 3
sbt "run --level 2.25"

# Almost fully level 3
sbt "run --level 2.95"
```

**Use cases:**
- Creating smooth level transitions in animations
- Visualizing the generation process
- Artistic effects with partial transparency

---

## Materials and Lighting

### Material Presets

Menger includes several physically-based material presets for OptiX mode:

```bash
# Glass materials
material=glass               # Standard glass (IOR 1.5)
material=water               # Clear water (IOR 1.33)
material=diamond             # Crystal clear diamond (IOR 2.42)

# Metallic materials
material=chrome              # Shiny metallic chrome
material=gold                # Metallic gold with yellow tint
material=copper              # Metallic copper with orange tint
material=metal               # Generic brushed metal

# Matte materials
material=plastic             # Matte plastic
material=matte               # Non-reflective matte surface

# Translucent / interference materials
material=film                # Thin-film interference (500 nm default; iridescent rainbow colors)
material=parchment           # Translucent paper-like surface
```

**Thin-Film Interference (`film-thickness=NM`):**

The `film` preset and any material can have a `film-thickness` parameter (nanometers) that enables
physically-based thin-film interference — the same optical effect as soap bubbles and oil slicks.
The color shifts with both thickness and viewing angle (angle of incidence).

```bash
# Film preset with default thickness (500 nm → green constructive interference)
--objects 'type=sphere:material=film'

# Explicit thickness values (different dominant colors)
--objects 'type=sphere:material=film:film-thickness=300'   # ~violet/blue
--objects 'type=sphere:material=film:film-thickness=500'   # ~green
--objects 'type=sphere:material=film:film-thickness=700'   # ~red/orange

# Oil-slick coating over another material
--objects 'type=sphere:material=chrome:film-thickness=400'
```

| Thickness (nm) | Dominant color at normal incidence |
|---------------|------------------------------------|
| 200–300       | Violet / blue                      |
| 400–500       | Green                              |
| 550–650       | Yellow / orange                    |
| 650–750       | Red                                |

**Usage with `--objects` flag:**
```bash
sbt "run --optix \
    --objects 'type=sphere:material=glass:pos=0,0,0' \
    --objects 'type=cube:material=gold:pos=2,0,0'"
```

### Custom Materials

Create custom materials by specifying individual parameters:

```bash
# Index of refraction (IOR) via --objects
--objects type=sphere:ior=1.5      # Glass
--objects type=sphere:ior=1.33     # Water
--objects type=sphere:ior=2.42     # Diamond
--objects type=sphere:ior=1.0      # Opaque (no refraction)

roughness=0.1                # Surface roughness (0.0-1.0)
metallic=0.8                 # Metallic property (0.0-1.0)
specular=0.5                 # Specular reflection intensity (0.0-1.0)
```

**Material Properties:**

| Property | Range | Effect |
|----------|-------|--------|
| `ior` | 1.0-3.0 | Index of refraction (1.0=opaque, higher=more refraction) |
| `roughness` | 0.0-1.0 | Surface roughness (0.0=mirror, 1.0=matte) |
| `metallic` | 0.0-1.0 | Metallic vs dielectric (0.0=plastic, 1.0=metal) |
| `specular` | 0.0-1.0 | Specular reflection intensity |
| `color` | #RRGGBB[AA] | Base color with optional alpha |

**Examples:**
```bash
# Custom glass with higher roughness (frosted glass)
sbt "run --optix --objects 'type=sphere:ior=1.5:roughness=0.3'"

# Brushed metal sphere
sbt "run --optix --objects 'type=sphere:metallic=1.0:roughness=0.5'"

# Semi-transparent colored sphere
sbt "run --optix --objects 'type=sphere:color=#FF000080:ior=1.3'"
```

### Lighting Setup

#### Adding Lights

OptiX mode supports up to 8 light sources. Use the `--light` flag (repeatable):

```bash
--light <type>:<x,y,z>[:intensity[:color]]
```

**Light Types:**

**Directional Light** (`directional`)
- Parallel rays (like sunlight)
- Direction vector points TOWARD the light source (where light comes from)
  - Light rays travel in opposite direction, shining onto the scene
  - Example: `directional:1,-1,-1` places light at upper-right-back
- Normalized automatically
- No falloff with distance
- Example: `--light directional:-1,1,-1`

**Point Light** (`point`)
- Radiates from a position in all directions
- Intensity falls off with distance (inverse square law)
- Example: `--light point:0,5,0:2.0`

**Area Light** (`area`)
- Disk emitter — produces soft shadows with visible penumbra
- Configurable shadow sample count per light (1–16, default 4)
- Format: `area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color]]]`
  - `px,py,pz` — center position of the disk
  - `nx,ny,nz` — normal direction (toward the scene), auto-normalized
  - `radius` — disk radius in world units
  - `samples` — shadow rays per light (default 4; more = softer but slower)
- Example: `--light area:0,4,0:0,-1,0:1.5:8`  (disk above, facing down, 8 samples)

**Light Parameters:**

```bash
# Basic directional light from upper-left
--light directional:-1,1,-1

# Bright point light above the scene
--light point:0,5,0:2.0

# Red directional light
--light directional:0,1,0::ff0000

# Colored point light with intensity
--light point:2,3,2:1.5:ffd700    # Gold-colored light

# Area light: disk above the scene, 4 shadow samples (default)
--light area:0,4,0:0,-1,0:1.5

# Area light with 8 shadow samples for softer penumbra
--light area:0,4,0:0,-1,0:1.5:8

# Combine area light with shadows for soft shadow rendering
--shadows --light area:0,4,0:0,-1,0:2.0:8
```

#### Multi-Light Setup

```bash
# Three-point lighting setup
sbt "run --optix --objects 'type=sphere' \
    --light directional:-1,1,-1:1.5 \          # Key light
    --light directional:1,0.5,-1:0.5:8080ff \  # Fill light (blue)
    --light point:0,3,2:0.8:ffffff"             # Rim light
```

#### Shadows

Enable shadow rays for realistic shadows:

```bash
sbt "run --optix --objects 'type=sphere' \
    --shadows \
    --light directional:-1,1,-1"
```

**Note:** Shadows increase render time but add significant realism.

#### Transparent Shadows

By default, transparent (glass) objects cast opaque gray shadows proportional to their alpha.
Enable `--transparent-shadows` to make glass objects cast color-tinted shadows instead — a red
glass sphere casts a red-tinted shadow, a blue glass sphere a blue-tinted shadow:

```bash
sbt "run --optix \
    --objects 'type=sphere:color=#FF000066:ior=1.5' \
    --shadows --transparent-shadows \
    --light directional:-1,1,-1"
```

**How it works:** The shadow ray records the transparent object's color and opacity and
attenuates each RGB channel of the light independently. A red sphere with alpha 0.6 passes
100% of the red channel but only 40% of green and blue, creating a red-tinted shadow.

**Multi-object accumulation (Phase 2):** Shadow rays accumulate color tinting through all
transparent objects multiplicatively. Two overlapping glass spheres — one red, one blue — cast
a combined purple shadow. An opaque object behind them still casts a fully dark shadow
(the opaque closest-hit overwrites the accumulated tint).

**Limitation:** Transparent shadows through **cylinder** objects may double-count opacity because
a ray can intersect both the cylinder body and its end caps. Use transparent cylinders with caution;
sphere and mesh objects behave correctly.

#### Default Lighting

If no lights are specified, a default setup is used:
- One directional light from upper-left
- Ambient light for basic visibility

---

### Plane Materials

The ground plane supports a material preset in addition to the basic color/checkerboard options:

```bash
# Solid color plane (original behavior)
sbt "run --optix --plane-color 808080"

# Checkerboard plane (original behavior)
sbt "run --optix --plane-color ffffff:404040"

# Material preset plane (new in Sprint 13)
sbt "run --optix --plane-material chrome"   # Mirror-finish chrome floor
sbt "run --optix --plane-material gold"     # Gold-tinted floor
```

**Available plane material presets:** `glass`, `water`, `diamond`, `chrome`, `gold`, `copper`,
`metal`, `plastic`, `matte`, `film`, `parchment`

**Note:** `--plane-material` and `--plane-color` are mutually exclusive — use one or the other.

---

### Material Reference

Reference values for physically plausible materials. The DSL's `Material.validate()` method
returns advisory warnings for combinations that deviate from these ranges.

| Material | IOR | Roughness | Metallic | Notes |
|----------|-----|-----------|----------|-------|
| Glass | 1.45–1.90 | 0.0–0.1 | 0.0 | Never metallic |
| Water | 1.33 | 0.0–0.05 | 0.0 | Very smooth |
| Diamond | 2.42 | 0.0 | 0.0 | High IOR |
| Chrome | 1.0 | 0.0–0.2 | 1.0 | Always metallic |
| Gold | 1.0 | 0.0–0.3 | 1.0 | Always metallic |
| Copper | 1.0 | 0.0–0.3 | 1.0 | Always metallic |
| Plastic | 1.4–1.6 | 0.3–0.7 | 0.0 | Never metallic |
| Matte | 1.0 | 0.8–1.0 | 0.0 | Diffuse only |

#### Mixed Metallic Values

While most real-world materials are either fully metallic or fully dielectric, partial metallic
values (0 < metallic < 1) are valid in PBR and produce interesting hybrid appearances:

| `metallic` | Visual character |
|------------|-----------------|
| 0.0 | Pure dielectric (plastic-like) |
| 0.25 | Slightly metallic sheen |
| 0.5 | Equal mix — painted metal, patina |
| 0.75 | Mostly metallic, softened |
| 1.0 | Pure metal |

```bash
# Gradient metallic showcase (DSL)
val objects = (0 to 4).map { i =>
  val m = i * 0.25f
  Sphere(pos = Vec3(i * 2.5f - 5f, 0f, 0f), material = Material(metallic = m, roughness = 0.3f))
}
```

See the `MixedMetallicShowcase` example scene for a ready-to-run demonstration.

#### Validation Warnings (DSL)

`Material.validate()` returns advisory strings for suspicious combinations:

```scala
val mat = Material(ior = 1.5f, metallic = 1.0f)
mat.validate().foreach(println)
// "Metallic materials typically use IOR=1.0 in PBR, got IOR=1.5"
```

Warnings are advisory — they do not prevent rendering. Use them to catch accidental
misconfigurations in scene scripts.

---

← [Quick Start](quickstart.md) | [User Guide Index](../USER_GUIDE.md) | → [Advanced Features](advanced.md)
