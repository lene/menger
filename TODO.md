# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or sprint plans as appropriate.

## Obsolete — Handled by t-Parameter Animation System (Sprint 12)

The `scene(t)` function introduced in Sprint 12 renders these features unnecessary.
Return different values from `scene(t)` to achieve any of these effects in plain Scala.

- ~~Camera animation (path following)~~ — return different `Camera` from `scene(t)`
- ~~Light animation~~ — return different `lights` from `scene(t)`
- ~~Property animation (colors, IOR, etc.)~~ — rebuild objects with new params each frame
- ~~Rotation / full object transform animation (position, rotation, scale)~~ — rebuild with new `Vec3` each frame
- ~~Keyframe system with linear interpolation~~ — lerp in Scala inside `scene(t)`
- ~~DSL animation syntax (deferred from Sprint 10)~~ — plain Scala IS the DSL
- ~~Easing functions (ease-in-out, cubic, bounce, elastic)~~ — implement in Scala in `scene(t)`

## Scheduled

Items linked to their sprint. See [ROADMAP.md](ROADMAP.md) for details.

- texture/material for plane → **Sprint 13 (13.1)**
- shadows with transparent objects → **Sprint 13 (13.2)**
- check materials for real-life correctness → **Sprint 13 (13.3)**
- examples for mixed-metallic (0 < metallic < 1) materials → **Sprint 13 (13.4)**

- Video output via ffmpeg → **Sprint 14 (14.1)**
- animation preview (interactive t scrubbing) → **Sprint 14 (14.2)**
- Soft shadows / area lights → **Sprint 14 (14.3)**
- Depth of field / bokeh → **Sprint 14 (14.4)**
- Additional primitives: cylinder, cone, torus → **Sprint 14 (14.5)**
- Coordinate cross / axis visualization → **Sprint 14 (14.6)**

- optimize pre-push hook execution time by parallelization → **Sprint 15 (15.7)**
- better agent instructions for updating documentation and changelog → **Sprint 15 (15.8)**
- better developer documentation → **Sprint 15 (15.8)**
- Runtime scene evaluation (currently compile-time only) → **Sprint 15 (15.5)**
- Window/output settings in DSL (width, height, saveName, headless) → **Sprint 15 (15.1)**
- Scene composition helpers and utilities → **Sprint 15 (15.2)**
- Procedural placement/generation helpers in DSL → **Sprint 15 (15.3)**
- Bezier/spline camera path utility → **Sprint 15 (15.4)**
- Animation export/import (JSON format for t-param configs) → **Sprint 15 (15.6)**

- 4D and 3D sponge cutaways → **Sprint 16 (16.1)**
- other polytopes in 3D (octahedron, dodecahedron, icosahedron) → **Sprint 16 (16.2)**
- other polytopes in 4D (16-cell, 24-cell, 600-cell) → **Sprint 16 (16.3)**
- parametrized surfaces in 3D → **Sprint 16 (16.4)**
- better user guide documentation → **Sprint 16 (16.5) + Sprint 17 (17.7)**

- images as backgrounds / environment maps / skybox → **Sprint 17 (17.1)**
- do we even support procedural textures? If not, do so → **Sprint 17 (17.2)**
- Add more procedural textures (wood, marble, etc.) → **Sprint 17 (17.3)**
- sponge with xyz → RGB mapping procedural texture → **Sprint 17 (17.4)**
- PBR texture support (normal + roughness maps) → **Sprint 17 (17.5)**
- test on both CUDA 12 and 13 (docker images for both) → **Sprint 17 (17.6)**

- CUDA 4D transform and projection → **Sprint 18 (18.1)**
- parametrized surfaces in 4D → **Sprint 18 (18.2)**
- create a website with feedback button (opening GitHub or GitLab issue pre-filled with template) → **Sprint 18 (18.3)**

- higher dimensional menger sponge and sierpinski tetrahedron analogs → **Sprint 19 (19.1, 19.2)**

## Unscheduled

Ideas not yet assigned to a sprint.

- multiple planes, as well as zero
- DSL syntax for render settings beyond those currently supported
- Guidance for generating good and interesting scenes and animations (user guide)

## Long-Term Backlog (Sprint 20+)

- L-systems in 3D and 4D
- rotopes for higher-dimensional geometry generation
- stereoscopic 3D rendering (left/right eye cameras)
