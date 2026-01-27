# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or [SPRINT.md](docs/sprints/SPRINT.md) as appropriate.

- fix PushToGithub (see https://gitlab.com/lilacashes/menger/-/jobs/12865767482)
- find better names for sponge-2 and tesseract-sponge-2
- validate tesseract sponge generation from surfaces by repeating it with cubes. same result?
- parchment has an ior, it shouldn't. instead it should attenuate light like... well, parchment
- ior (and maybe others) are both part of ObjectSpec and separate CLI parameter - redundant
- texture/material for plane
- check materials for real-life correctness - find references
- ~~generalize TesseractEdgeSceneBuilder to support other 4D objects~~ → **Sprint 9.9**
- Add more procedural textures (wood, marble, etc.) as a texture library → **Deferred to future sprint**
- do we even support procedural textures?
- better user guide documentation
- better developer documentation
- better agent instructions for updating documentation and changelog, monitoring ci pipelines after
  push, using glab (what else is missing?)
- sponge with xyz -> RGB mapping procedural texture → **Deferred (needs procedural texture infrastructure)**
- Scala wrapper for libGDX to delegate var and null usage to that layer
- shadows with transparent objects
- 4D and 3D sponge cutaways → **Deferred to Sprint 10 or 11**
- ~~4D and 3D objects in a scene together~~ → **Sprint 9.10 (verification)**
- 4D camera distance with shift mouse wheel → **Deferred to Sprint 10 or 11**
- examples for mixed-metallic (0 < metallic < 1) materials
- fractional sponges in OptiX → **Sprint 9 INCOMPLETE** - only floors level, missing transparency overlay
  - Need to implement dual-object rendering: level L+1 (opaque) + level L (transparent)
  - See `FractionalRotatedProjection.scala` for LibGDX reference implementation
- 4D rotation with Shift+arrow keys → **Sprint 9 INCOMPLETE** - handler exists but not called
  - Fix: add `keyHandler.update(Gdx.graphics.getDeltaTime)` to `OptiXEngine.render()`
  - Shift+mouse drag works, only keyboard is broken
- parametrized surfaces in 3D and 4D
- multiple planes, as well as zero
- Rounded edges on cubes/sponges
