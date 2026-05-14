# Sprint 19: Advanced Geometry

**Sprint:** 19 - Advanced Geometry
**Status:** Complete
**Estimate:** ~30h
**Branch:** `feature/sprint-19`
**Dependencies:** Sprint 18 (multi-GAS IAS, IS programs, GPU 4D math)

---

## Goal

Expand the geometry system with additional polytopes in 3D and 4D, analytical primitives,
planes as first-class geometry, a coordinate cross visualization, and a geometry registry
for extensibility.

## Success Criteria

- [x] Tetrahedron, octahedron, dodecahedron, icosahedron available as 3D primitives
- [x] Pentachoron (5-cell), 16-cell, 24-cell, 120-cell, and 600-cell available as 4D primitives (standalone, no subdivision)
- [x] Cone available as analytical primitive (IS program from Sprint 18); torus cancelled — tesselated version remains
- [x] Planes are first-class scene geometry with materials (not miss shader)
- [x] Coordinate cross / axis visualization works alongside other geometry
- [x] Geometry registry: adding a new type requires registration, not engine modification
- [x] Per-object 3D rotation (X/Y/Z) works via scene graph and CLI
- [x] Render time per frame and per ray reported in stats output
- [x] Spike findings documented for max trace depth
- [x] Spike findings documented for fractional IAS sponges
- [x] All tests pass

---

## Tasks

### Task 19.1: Additional Polytopes in 3D

**Estimate:** 4h

Add tetrahedron, octahedron, dodecahedron, and icosahedron as first-class primitives in
the DSL and rendering engine, alongside the existing cube and sphere. None of these are
currently implemented.

Implemented as triangle meshes via the existing `Builder` trait.

---

### Task 19.2: Additional Polytopes in 4D

**Estimate:** 6h
**Depends on:** Sprint 18.3 (GPU 4D math)

Add pentachoron (5-cell), 16-cell, 24-cell, 120-cell, and 600-cell as 4D primitives.
None of these are currently implemented. These are rendered using the GPU 4D math from
Sprint 18.3, avoiding the legacy CPU projection path.

**Scope:** Standalone polytopes only (rotate, project, view). Fractal subdivision on these
bases is deferred to a later sprint.

**Notes:**
- The pentachoron (5-cell) is the 4D analog of the tetrahedron: 5 vertices, 10 edges, 10 triangular faces, 5 tetrahedral cells.
- The 120-cell has 600 vertices and 120 dodecahedral cells; the 600-cell has 120 vertices and 600 tetrahedral cells. Both are compute-heavy at higher rotation counts — a level guard will be added if used as subdivision bases.

---

### Task 19.3: Analytical Primitives — Cone (Torus cancelled)

**Estimate:** 2h
**Depends on:** Sprint 18.2 (IS program infrastructure)

Add cone as analytical primitive using a custom OptiX intersection program.
The analytical torus was cancelled due to intractable normal-sign issues in the
quartic solver; the tesselated `ParametricTorus` remains available.

**Torus cancellation rationale:** The Schwarze quartic solver produced correct
intersection distances but the surface normals were inverted (geometric normals
pointed outward from the solid, but the renderer treated them as inward-facing).
Despite extensive debugging (gradient-based normals, root validation), the sign
could not be corrected. The tesselated torus (`ParametricTorus`) renders correctly
and is the recommended approach.

DSL usage:
```scala
scene(
  cone(height = 2.0, radius = 1.0, material = gold)
)
```

Tessellated versions remain available as `TessellatedCone`, `TessellatedTorus`.

---

### Task 19.4: Planes as First-Class Geometry

**Estimate:** 2h
**Depends on:** Sprint 18.1 (multi-GAS IAS), Sprint 18.2 (IS programs)

Promote planes from the miss shader to proper scene geometry with:
- Custom IS program for plane intersection
- Per-plane material (inheritable via scene graph)
- Position and normal as parameters
- Zero or more planes per scene

The existing miss shader becomes a solid color or environment map background (Sprint 20).

---

### Task 19.5: Coordinate Cross (Axis Visualization)

**Estimate:** 1.5h
**Depends on:** Sprint 18.1 (multi-GAS IAS), Sprint 18.2 (IS programs)

Render XYZ axis lines for debugging scene layout and camera positioning.

- Thin analytical cylinders along X (red), Y (green), Z (blue) axes
- Configurable length and thickness
- Toggle on/off via CLI or keyboard shortcut

---

### Task 19.6: Geometry Registry

**Estimate:** 2h
**Depends on:** Sprint 17.3 (scene graph)

Replace the match expression in `OptiXEngine.selectMeshBuilder()` with a registration-based
`GeometryRegistry`. Adding a new geometry type means registering it (name, builder factory,
IS program if analytical, DSL constructor), not modifying the engine.

Scene graph nodes reference geometry by registered type. The registry provides:
- Name → builder/IS program mapping
- CLI option generation for `--objects` syntax
- DSL constructor availability

---

### Task 19.7: Per-Object 3D Rotation via Scene Graph

**Estimate:** 3h
**Depends on:** Sprint 17.3 (scene graph)

Expose per-object 3D rotation (X, Y, Z Euler angles or axis-angle) through the scene
graph `Transform` node and the CLI `--objects` syntax. This is the primary reason the
scene graph was introduced — objects should be positionable and orientable in world space
without hardcoding transforms in geometry constructors.

**What changes:**
- `Transform` already carries translation, rotation, scale; verify rotation is wired
  through to the OptiX instance transform matrix.
- CLI: `--objects type=sphere:rotX=30:rotY=45:rotZ=0` (degrees).
- DSL: `rotate(x=30.deg, y=45.deg)` wrapper on `SceneNode`.
- Integration test: two cubes at different orientations produce distinguishable renders.

**Success criterion:** A cube rotated 45° around Y looks like a rotated cube, not an
axis-aligned one.

---

### Task 19.8: Render Time Stats per Frame and per Ray

**Estimate:** 1h

Add GPU render time (ms/frame) and estimated ms/ray to the stats output printed after
each headless render and displayed in the interactive overlay. The primary ray count
is already tracked; divide elapsed GPU time by ray count.

**What changes:**
- Record wall time around the OptiX launch call in `OptiXRenderer`.
- Compute ms/frame and ms/Mray; include in `RenderStats`.
- Print in CLI summary and interactive HUD.
- Unit test: `RenderStats` arithmetic is correct.

---

### Task 19.9: Spike — Max Trace Depth Above 8

**Estimate:** 1h (spike only — no feature commitment)

Investigate whether `MAX_TRACE_DEPTH` can be raised beyond 8 without requiring an
OptiX pipeline stack-size change. Document findings:
- What is the OptiX limit?
- What stack size does each additional depth level require?
- Is there a visible quality difference beyond depth 8 for glass stacks?
- Recommendation: raise to N, or accept 8 as the practical ceiling?

Output: a short findings note added to `docs/dev/` and a recommendation for whether
to schedule a follow-up task.

---

### Task 19.10: Spike — Fractional Levels with IAS Sponges

**Estimate:** 2h (spike only — no feature commitment)

Investigate whether fractional sponge levels (alpha-blended transition between integer
levels) can be applied to the recursive IAS sponge (`sponge-recursive-ias`). The
tessellated sponge already supports fractional levels via vertex alpha. Key questions:
- Can IAS instances be alpha-blended at the instance level, or does it require
  per-face vertex alpha in the leaf GAS?
- Is the alpha-blend approach compatible with the O(N·20) VRAM advantage?
- What is the implementation complexity vs. the tessellated approach?

Output: findings note in `docs/dev/` and a recommendation for Sprint 21.

---

### Task 19.11: Documentation

**Estimate:** 2h

- Sprint retrospective
- CHANGELOG.md update
- `docs/guide/user-guide.md` geometry section updated
- Example scenes for each new primitive
- arc42 update: geometry registry architecture

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 19.1 | Additional polytopes in 3D (tetrahedron + 3 others) | 4h | None |
| 19.2 | Additional polytopes in 4D (pentachoron + 4 others) | 6h | Sprint 18.3 |
| 19.3 | Analytical primitives: cone (torus cancelled) | 2h | Sprint 18.2 |
| 19.4 | Planes as first-class geometry | 2h | Sprint 18.1, 18.2 |
| 19.5 | Coordinate cross / axis visualization | 1.5h | Sprint 18.1, 18.2 |
| 19.6 | Geometry registry | 2h | Sprint 17.3 |
| 19.7 | Per-object 3D rotation via scene graph | 3h | Sprint 17.3 |
| 19.8 | Render time stats per frame and per ray | 1h | None |
| 19.9 | Spike: max trace depth above 8 | 1h | None |
| 19.10 | Spike: fractional levels with IAS sponges | 2h | Sprint 18.4 |
| 19.11 | Documentation | 2h | All |
| **Total** | | **~26.5h** | |

---

## Definition of Done

- [x] All success criteria met
- [x] All tests passing
- [x] Code quality checks pass: `sbt "scalafix --check"`
- [x] CHANGELOG.md updated
- [x] `docs/guide/user-guide.md` geometry section updated
- [ ] Example scenes created for each new primitive (deferred: new types CLI-only, no DSL case classes yet)
- [x] Geometry registry documented for contributors

---

## Notes

### Formerly

This sprint combines content from the original Sprint 18 (Advanced Geometry) with
planes as first-class geometry and the geometry registry. The original Sprint 18 content
(multi-GAS IAS, IS infrastructure, GPU 4D math) has moved to the new Sprint 18 as
infrastructure prerequisites.

### Deferred

- **Fractal subdivision on 4D polychora** — deferred to Sprint 21+ (standalone rendering first)
- **Sponge cutaways** — remains in backlog (clipping-plane infrastructure non-trivial)
- **Schläfli polytope generator** — backlog (algorithmic construction from `{p,q}` symbols)
- **Cylinder as DSL primitive** — TBD based on whether the analytical cylinder IS program
  from the coordinate cross covers the use case
