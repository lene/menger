# Sprint 36: Data Visualization I

**Sprint:** 36 - Data Visualization I
**Status:** Not Started
**Estimate:** ~25 hours
**Branch:** `feature/sprint-36`
**Dependencies:** None

---

## Goal

Add data visualization capabilities: scalar-to-color mapping (colormaps), GPU evaluation
of scalar fields `f(x,y,z)`, isosurface rendering via sphere tracing, and basic volume
rendering via ray marching. Foundation for future dataset import (later sprint).

---

## Success Criteria

- [ ] Built-in colormaps (viridis, plasma, jet, grayscale) applicable to any geometry
- [ ] `ScalarField { fn = "(x,y,z) => sin(x)*cos(y)*z", isovalue = 0.5 }` renders isosurface
- [ ] Volume rendering: density field visualised as fog-like volume
- [ ] All features DSL-accessible
- [ ] Integration test reference images for each feature

---

## Tasks

### Task 36.1: Color by Intensity / Colormaps

**Estimate:** 5h

Scalar-to-color mapping for geometry: map a scalar value (distance, height, curvature,
or user-supplied) to a colour using a built-in colormap.

**DSL:**
```scala
addSphere(...) {
  colormap = Colormap.Viridis
  colormapSource = ColormapSource.Height  // or Distance, Normal, Custom
  colormapRange = (0f, 1f)
}
```

**Implementation:**
- Colormap as 1D CUDA texture (256 entries, uploaded once)
- In hit shader: compute scalar value from hit point properties; sample colormap texture
- Built-in maps: Viridis, Plasma, Jet, Grayscale (store as float4 arrays in C++)
- JNI: `setColormap(id: Int, data: Array[Float])`

---

### Task 36.2: Scalar Field GPU Evaluation

**Estimate:** 8h

Evaluate `f(x,y,z) → Float` on the GPU and render the isosurface via sphere tracing.

**DSL:**
```scala
addScalarField {
  expression = "sin(x * 3.14) * cos(y * 3.14) * z"  // parsed CPU-side, JIT to PTX
  isovalue = 0.0f
  bounds = AABB(Vec3(-2,-2,-2), Vec3(2,2,2))
  stepSize = 0.01f
  material { color = white; roughness = 0.3 }
}
```

**Implementation options (spike in 36.2 — pick one):**
1. Pre-evaluate on CPU grid → upload 3D texture → sphere trace against texture
2. Embed expression as inline CUDA device function (requires PTX recompilation)
3. Interpret a small expression bytecode on GPU

Option 1 is simplest: evaluate on CPU at scene load, upload 3D CUDA texture, sphere
trace in a custom intersection shader.

**Isosurface sphere tracing:** step along ray until sign change in `f(x) - isovalue`;
bisect for accurate hit point.

---

### Task 36.3: Volume Rendering

**Estimate:** 8h

Ray march through a scalar density field and accumulate colour/opacity.

**DSL:**
```scala
addVolume {
  expression = "exp(-x*x - y*y - z*z)"  // Gaussian blob
  bounds = AABB(Vec3(-2,-2,-2), Vec3(2,2,2))
  densityScale = 1.0f
  colormap = Colormap.Plasma
  stepSize = 0.05f
}
```

**Implementation:**
- Custom miss/hit integration: ray march through AABB, accumulate
  `color += transmittance * density * colormap(density) * stepSize`
- Early termination when transmittance < 0.01
- 3D texture (option 1 from 36.2) as density source

---

### Task 36.4: Tests + Documentation

**Estimate:** 4h

- Integration tests: colormap sphere, Gaussian isosurface, Gaussian volume
- `scripts/manual-test.sh` entries for visual review
- User guide: Data Visualization section (colormaps, scalar fields, volumes)
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 36.1 | Color by intensity / colormaps | 5h |
| 36.2 | Scalar field GPU evaluation + isosurface | 8h |
| 36.3 | Volume rendering (ray marching) | 8h |
| 36.4 | Tests + documentation | 4h |
| **Total** | | **~25h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated

---

## Notes

### Multi-dimensional Parameter Exploration

"Independently vary 2-3+ fractal parameters in real-time" (from backlog) is a UI/interaction
feature, not a visualization feature. Deferred to a future interaction-focused sprint.

### Dataset Import (VTK/NetCDF)

Deferred to a later sprint. Sprint 36 establishes the GPU volume/field infrastructure that
dataset import will build on.
