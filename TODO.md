# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or sprint plans as appropriate.

- ~~Scala DSL for scene description (type-safe, compiled with project)~~ → **Completed in Sprint 10 (v0.5.0)**
- ~~Scene files can import other files~~ → **Completed in Sprint 10 (v0.5.0)**
- check output - seems buggy (e.g. gold is transparent, fractional sponge level weirdnesses)
- coverage uses old coverage percentage as reference
- rotation of objects in 3D, possibly full transform (position, rotation, scale)
- Camera animation (path following)
- Light animation
- Property animation (colors, IOR, etc.)
- implement proper thin-film physics with thickness parameter and interference effects (Film material)
- validate tesseract sponge generation from surfaces by repeating it with cubes. same result?
- PBR texture support
- images as backgrounds
- Add more procedural textures (wood, marble, etc.) as a texture library
- do we even support procedural textures? If not, do so
- better user guide documentation
    - Update all examples to use --objects syntax (remove outdated --object, --radius, --ior, --scale, --center examples)
    - Guidance for generating good and interesting scenes and animations
- better developer documentation
- better agent instructions for updating documentation and changelog, monitoring ci pipelines after
  push, using glab (what else is missing?)
- sponge with xyz -> RGB mapping procedural texture (needs procedural texture infrastructure)
- Scala wrapper for libGDX to delegate var and null usage to that layer
- 4D and 3D sponge cutaways
- parametrized surfaces in 3D and 4D
- multiple planes, as well as zero
- test on both CUDA 12 and 13 (docker images for both)

- Shift+Scroll for 4D projection adjustment
- ESC to reset 4D view
- CLI shortcuts: `--4d-rotation`, `--4d-preset`
- State persistence (save/load 4D view)

- texture/material for plane → **Sprint 12 (Task 12.1)**
- check materials for real-life correctness → **Sprint 12 (Task 12.3)**
- shadows with transparent objects → **Sprint 12 (Task 12.2)**
- examples for mixed-metallic (0 < metallic < 1) materials → **Sprint 12 (Task 12.4)**
- Rounded edges on cubes/sponges → **Sprint 12 (Task 12.5, stretch goal)**

- Object transform animation (position, rotation, scale)
- Keyframe system with linear interpolation
- Frame sequence rendering
- DSL animation syntax (deferred from Sprint 10 - requires keyframe system first)
- DSL support for 4D objects (tesseract, tesseract-sponge) - deferred from Sprint 10

- Easing functions (ease-in-out, cubic, bounce, elastic)
- Video output via ffmpeg
- other polytopes in 3D and 4D
- higher dimensional menger sponge and sierpinski tetrahedron analogs 
- steroscopic 3D rendering (left/right eye cameras)
- create a website with feedback button (opening GitHub or GitLab issue pre-filled with template)

## DSL Deferred Features (from Sprint 10)

- Runtime scene evaluation (currently compile-time only)
- Window/output settings in DSL (width, height, saveName, headless — CLI-only for now)
- DSL syntax for render settings beyond those currently supported
- Scene composition helpers and utilities
- Procedural placement/generation helpers in DSL
