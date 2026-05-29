# Sprint 28: Visual Quality

**Sprint:** 28 - Visual Quality
**Status:** Not Started
**Estimate:** ~20 hours
**Branch:** `feature/sprint-28`
**Dependencies:** None

---

## Goal

Add two visual quality features: depth of field (camera aperture / bokeh blur) and
wireframe rendering (stylistic edge-only display). Both are self-contained rendering
pipeline additions with DSL integration.

---

## Success Criteria

- [ ] `camera { aperture = 0.1, focalDistance = 5.0 }` in DSL produces bokeh blur
- [ ] Depth of field disabled by default (aperture = 0 → pinhole camera, current behaviour)
- [ ] `material { wireframe = true }` renders mesh edges only as thin cylinders
- [ ] Both features have integration test reference images
- [ ] All tests pass

---

## Tasks

### Task 28.1: Depth of Field

**Estimate:** 8h

Physically-based depth of field via lens sampling in the raygen shader.

**DSL:**
```scala
case class Camera(
  eye: Vec3 = ...,
  lookAt: Vec3 = ...,
  up: Vec3 = ...,
  fovDegrees: Float = 60f,
  aperture: Float = 0f,       // 0 = pinhole (no DoF)
  focalDistance: Float = 1f,  // world units to focal plane
)
```

**Implementation:**
- In raygen shader: if `aperture > 0`, jitter ray origin on a disk of radius `aperture`
  perpendicular to view direction; adjust direction to converge at `focalDistance`
- Standard thin-lens approximation (PBRT §6.2)
- JNI: `setCameraDoF(aperture: Float, focalDistance: Float)`
- DSL: wire `Camera.aperture` and `Camera.focalDistance` through `SceneConfigurator`

**Integration test:** sphere scene with aperture=0.05, focal distance at sphere centre —
foreground and background blur visible.

---

### Task 28.2: Wireframe Rendering

**Estimate:** 6h

Stylistic wireframe via thin cylinders placed on triangle mesh edges.

**Implementation:**
- Post-process the triangle mesh: extract unique edges, add a thin cylinder (radius ~0.002
  world units) for each edge via the existing cylinder geometry path
- `material { mode = Wireframe }` triggers this in SceneConfigurator
- Optional: include face rendering at low opacity alongside wireframe

**DSL:**
```scala
enum RenderMode:
  case Solid      // default
  case Wireframe  // edges only
  case SolidWireframe  // faces + edges overlay

// Usage:
addMesh("model.obj") { renderMode = RenderMode.Wireframe }
```

**Integration test:** cube or platonic solid in wireframe mode.

---

### Task 28.3: Integration Tests + Reference Images

**Estimate:** 3h

- DoF scene: `integration-tests.sh` scenario with aperture blur
- Wireframe scene: cube or icosahedron in wireframe mode
- Add both to `scripts/manual-test.sh`

---

### Task 28.4: Documentation

**Estimate:** 3h

- User guide: Depth of Field section — aperture, focal distance, relationship to
  sample count for noise reduction
- User guide: Wireframe Rendering section — render modes, performance note
- CHANGELOG.md entry

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 28.1 | Depth of field — lens sampling in raygen | 8h |
| 28.2 | Wireframe rendering — edge cylinders | 6h |
| 28.3 | Integration tests + reference images | 3h |
| 28.4 | Documentation | 3h |
| **Total** | | **~20h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
