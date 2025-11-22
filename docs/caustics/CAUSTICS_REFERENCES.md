# Caustics Reference Images and Scenes

> **Purpose:** Collection of reference materials for validating Progressive Photon Mapping (PPM)
> caustics implementation. Used for test ladder Step 8 (C8) - Reference Match validation.
>
> **Location:** `optix-jni/test-resources/caustics-references/`

## Overview

This document catalogs reference scenes and images from established renderers that can be used
to validate our caustics implementation. The goal is to achieve SSIM > 0.90 against a
known-good reference for our canonical test scene.

---

## 1. Canonical Test Scene (Primary Reference)

**Location:** `test-resources/caustics-references/renders/canonical-caustics.pbrt`

This is our **primary reference scene**, designed to exactly match the parameters defined in
[arc42 Section 10](../docs/arc42/10-quality-requirements.md#canonical-test-scene).

### Scene Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sphere** | | |
| Center | (0, 0, 0) | Origin |
| Radius | 1.0 | Unit sphere |
| IOR | 1.5 | Standard glass (dielectric) |
| **Floor** | | |
| Position | Y = -2.0 | Horizontal plane |
| Material | Diffuse gray (0.8) | High albedo for visibility |
| Size | 20 × 20 units | Large enough for caustic |
| **Light** | | |
| Type | Point light | Simple, predictable |
| Position | (0, 10, 0) | Directly above sphere |
| Intensity | 500 | Bright enough for clear caustic |
| **Camera** | | |
| Position | (0, 1, 4) | Above and behind |
| Look at | (0, 0, 0) | Origin (sphere center) |
| FOV | 45° | Standard perspective |
| **Rendering** | | |
| Resolution | 800 × 600 | Match our default render size |
| Integrator | SPPM | Stochastic Progressive Photon Mapping |
| Max depth | 32 | Sufficient for glass traversal |
| Photons/iter | 100,000 | Good caustic density |
| Samples | 256 | High quality |

### Expected Result

- **Caustic shape:** Circular bright spot
- **Caustic center:** (0, -2, 0) - directly below sphere
- **Caustic radius:** ~0.3 units (from thick lens formula)
- **Brightness:** Peak > 1.5× ambient floor

### Rendering Commands

```bash
# With PBRT v4
pbrt canonical-caustics.pbrt

# Convert EXR to PNG for comparison
imgtool convert canonical-caustics.exr canonical-caustics.png
```

---

## 2. Appleseed Cornell Box (Secondary Reference)

**Location:** `test-resources/caustics-references/appleseed/cornell-box-caustics.appleseed`

**Source:** [Appleseed Renderer](https://github.com/appleseedhq/appleseed) test scenes

### Scene Parameters

| Parameter | Value |
|-----------|-------|
| **Glass Sphere** | |
| IOR | **1.5** (matches our canonical scene) |
| Material | specular_btdf (pure glass) |
| Transmittance | 1.0 (clear glass) |
| **Chrome Sphere** | |
| Material | specular_brdf (perfect mirror) |
| **Rendering** | |
| Resolution | 512 × 512 |
| Spectrum mode | Spectral (physically accurate) |
| Caustics | Explicitly enabled (`enable_caustics="true"`) |
| Max bounces | Unlimited (-1) |
| Sample generator | Light tracing |

### Key Features

- Uses spectral rendering (physically accurate colors)
- Classic Cornell box setup (red/green walls, white floor/ceiling)
- Both reflective and refractive caustics (chrome + glass spheres)
- Well-documented reference in academic literature

### Why Useful

This scene validates that our IOR 1.5 glass produces similar caustics to a trusted renderer.
The Cornell box is the most widely-used validation scene in ray tracing research.

### Rendering

Requires Appleseed renderer:
```bash
appleseed.cli cornell-box-caustics.appleseed -o cornell-caustics.png
```

---

## 3. PBRT v4 Scenes

**Location:** `test-resources/caustics-references/pbrt/pbrt-v4-scenes/`

**Source:** [github.com/mmp/pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes)

### 3.1 LTE-Orb (Recommended)

**Files:**
- `lte-orb/lte-orb-rough-glass.pbrt` - Glass sphere with rough surface
- `lte-orb/lte-orb-simple-ball.pbrt` - Simple measured material

| Parameter | Value |
|-----------|-------|
| Resolution | 1200 × 1200 |
| Integrator | volpath (max depth 30) |
| Samples | 256-1024 |
| Light | Area light / HDR environment |

**Why Useful:** The LTE-Orb is specifically designed as a "useful tool for visualizing the
appearance of various materials." The spherical geometry matches our canonical scene.

### 3.2 Transparent Machines

**Files:**
- `transparent-machines/frame542.pbrt`
- `transparent-machines/frame812.pbrt`
- `transparent-machines/frame888.pbrt`

| Parameter | Value |
|-----------|-------|
| Material | Complex glass shapes |
| Max bounces | 64+ required |
| Light | Skylight (HDR environment) |

**Why Useful:** Tests complex glass caustics from multiple interacting surfaces.
Good for stress-testing the implementation.

### 3.3 Crown

**File:** `crown/crown.pbrt`

| Parameter | Value |
|-----------|-------|
| Materials | Gold metal + gems |
| Caustics | Refractive (gems) + reflective (gold) |

**Why Useful:** Tests gem-like refractive caustics (colored glass).

---

## 4. Henrik Jensen's Original Work

**Source:** Foundational photon mapping research by Henrik Wann Jensen

### Papers (Reference Only - Not Downloaded)

| Paper | URL | Key Images |
|-------|-----|------------|
| Global Illumination using Photon Maps (1996) | [Stanford PDF](https://graphics.stanford.edu/~henrik/papers/ewr7/egwr96.pdf) | Glass sphere caustic on floor |
| A Practical Guide to Global Illumination (2001) | [Princeton PDF](https://www.cs.princeton.edu/courses/archive/fall18/cos526/papers/jensen01.pdf) | Figure 18: caustics photon map |
| Realistic Image Synthesis Using Photon Mapping (Book) | [Book info](http://graphics.ucsd.edu/~henrik/papers/book/) | Comprehensive reference |

### Key Parameters from Jensen's Work

From the 1996 paper:
- "289,000 photons in the caustics photon map"
- "165,000 photons in the global photon map"
- Glass sphere with standard IOR (~1.5)
- Two small spherical area light sources

### Jensen's Canonical Description

> "Note the caustic below the glass sphere, the glossy reflections, and the overall
> quality of the global illumination."

---

## 5. Mitsuba 3 Tutorial

**Location:** `test-resources/caustics-references/mitsuba/caustics_optimization.ipynb`

**Source:** [Mitsuba Tutorials](https://github.com/mitsuba-renderer/mitsuba-tutorials)

### Purpose

This is an **inverse rendering** tutorial that recovers heightmap displacement to create
specific caustic patterns. While not directly a validation reference, it demonstrates:

1. Caustic formation through glass slabs
2. Quantitative comparison of rendered vs target caustics
3. Gradient-based optimization for caustic shapes

### Reference Images

The tutorial uses target caustic patterns:
- `wave-1024.jpg` - Wave pattern target
- `sunday-512.jpg` - Complex image target

### Why Useful

Demonstrates that caustics can be quantitatively compared and optimized toward a target,
validating our SSIM-based comparison approach.

---

## 6. Additional Resources

### Online Galleries

| Source | URL | Notes |
|--------|-----|-------|
| Henrik Jensen's Caustics Gallery | [cseweb.ucsd.edu/~henrik/images/caustics.html](https://cseweb.ucsd.edu/~henrik/images/caustics.html) | Original photon mapping results |
| Benedikt Bitterli's Resources | [benedikt-bitterli.me/resources/](https://benedikt-bitterli.me/resources/) | Free scenes in PBRT/Mitsuba/Tungsten |
| NVIDIA OptiX Samples | [github.com/NVIDIA/OptiX_Apps](https://github.com/NVIDIA/OptiX_Apps) | GPU ray tracing examples |

### Academic References

| Paper | Relevance |
|-------|-----------|
| "Progressive Photon Mapping" (Hachisuka et al., 2008) | Core PPM algorithm |
| "Stochastic Progressive Photon Mapping" (Hachisuka & Jensen, 2009) | SPPM variant used in PBRT |
| "Opposite Renderer" (Partridge thesis) | OptiX PPM implementation |

---

## Validation Workflow

### Step 1: Render Reference

```bash
# Option A: Use PBRT (recommended)
cd test-resources/caustics-references/renders
pbrt canonical-caustics.pbrt
imgtool convert canonical-caustics.exr canonical-caustics-reference.png

# Option B: Use Mitsuba
mitsuba canonical-caustics.xml -o canonical-caustics-reference.exr
```

### Step 2: Render Test Image

```bash
# Our implementation
./menger --object sphere --renderer optix --caustics --output test-caustics.png
```

### Step 3: Compare

```scala
// Scala test
val reference = ImageIO.read(new File("canonical-caustics-reference.png"))
val test = ImageIO.read(new File("test-caustics.png"))
val ssim = ImageComparison.ssim(reference, test)
ssim should be >= 0.90
```

---

## File Structure

```
test-resources/caustics-references/
├── appleseed/
│   └── cornell-box-caustics.appleseed    # Appleseed scene file
├── jensen/
│   └── (papers - reference URLs only)
├── mitsuba/
│   └── caustics_optimization.ipynb       # Mitsuba tutorial
├── pbrt/
│   └── pbrt-v4-scenes/
│       ├── crown/                        # Gem caustics
│       ├── lte-orb/                      # Sphere test scenes
│       └── transparent-machines/         # Complex glass
└── renders/
    └── canonical-caustics.pbrt           # Our primary reference scene
```

---

## Recommended Reference for C8 Validation

**Primary:** `renders/canonical-caustics.pbrt` rendered with PBRT v4

**Rationale:**
1. Exact parameter match to our canonical test scene
2. PBRT is the gold standard for physically-based rendering
3. SPPM integrator is designed for caustics
4. Same resolution (800×600) as our default
5. Well-documented, reproducible

**Backup:** Appleseed Cornell box (same IOR, trusted renderer)

---

## References

- [arc42 Section 10 - Quality Requirements](../docs/arc42/10-quality-requirements.md)
- [CAUSTICS_TEST_LADDER.md](./CAUSTICS_TEST_LADDER.md)
- [PBRT Book, 4th Edition](https://pbrt.org)
- Jensen, H.W. "Realistic Image Synthesis Using Photon Mapping" (2001)
