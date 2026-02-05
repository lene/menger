# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or [SPRINT.md](docs/sprints/SPRINT.md) as appropriate.

- parallelize integration tests
- ~~fix code quality ci job~~
- implement proper thin-film physics with thickness parameter and interference effects (Film material)
- find better names for sponge-2 and tesseract-sponge-2 (volume filling vs surface unfolding)
- validate tesseract sponge generation from surfaces by repeating it with cubes. same result?
- ~~parchment has an ior, it shouldn't. instead it should attenuate light like... well, parchment~~
- ior (and maybe others) are both part of ObjectSpec and separate CLI parameter - redundant
- texture/material for plane
- check materials for real-life correctness - find references
- PBR texture support
- images as backgrounds 
- Add more procedural textures (wood, marble, etc.) as a texture library → **Deferred to future sprint**
- do we even support procedural textures? If not, do so
- better user guide documentation
  - Guidance for generating good and interesting scenes and animations
- better developer documentation
- better agent instructions for updating documentation and changelog, monitoring ci pipelines after
  push, using glab (what else is missing?)
- sponge with xyz -> RGB mapping procedural texture → **Deferred (needs procedural texture infrastructure)**
- Scala wrapper for libGDX to delegate var and null usage to that layer
- shadows with transparent objects
- 4D and 3D sponge cutaways → **Deferred to future sprint**
- 4D camera distance with shift mouse wheel → **Deferred to Sprint 10**
- examples for mixed-metallic (0 < metallic < 1) materials
- parametrized surfaces in 3D and 4D
- multiple planes, as well as zero
- Rounded edges on cubes/sponges
- test on both CUDA 12 and 13 (docker images for both)
