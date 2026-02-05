# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or sprint plans as appropriate.

## Planned in Sprint 10 (Scala DSL)
- Scala DSL for scene description (type-safe, compiled with project)
- Block-style and case-class syntax
- Scene files can import other files
- CLI: `--scene scenes.MyScene`

## Planned in Sprint 11 (4D Framework Enhancements)
- Shift+Scroll for 4D projection adjustment
- ESC to reset 4D view
- CLI shortcuts: `--4d-rotation`, `--4d-preset`
- State persistence (save/load 4D view)

## Planned in Sprint 12 (Visual Quality & Materials)
- texture/material for plane → **Sprint 12 (Task 12.1)**
- check materials for real-life correctness → **Sprint 12 (Task 12.3)**
- shadows with transparent objects → **Sprint 12 (Task 12.2)**
- examples for mixed-metallic (0 < metallic < 1) materials → **Sprint 12 (Task 12.4)**
- Rounded edges on cubes/sponges → **Sprint 12 (Task 12.5, stretch goal)**

## Planned in Sprint 13 (Object Animation Foundation)
- Object transform animation (position, rotation, scale)
- Keyframe system with linear interpolation
- Frame sequence rendering
- DSL animation syntax

## Planned in Sprint 14 (Advanced Animation)
- Easing functions (ease-in-out, cubic, bounce, elastic)
- Camera animation (path following)
- Light animation
- Property animation (colors, IOR, etc.)
- Video output via ffmpeg

## Backlog (Not Yet Scheduled)
- implement proper thin-film physics with thickness parameter and interference effects (Film material)
- find better names for sponge-2 and tesseract-sponge-2 (volume filling vs surface unfolding)
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

## Recently Completed
- ✅ Fractional 3D sponge levels (Sprint 9)
- ✅ Fractional 4D sponge levels (Sprint 9)
- ✅ Interactive 4D rotation with Shift+Arrow keys (Sprint 8-9)
- ✅ Interactive 4D rotation with mouse drag (Sprint 8-9)
- ✅ Mesh4D abstraction (Sprint 8-9)
- ✅ Remove legacy CLI options (--radius, --ior, --scale, --center)
