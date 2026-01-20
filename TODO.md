# TODO

Quick notes and ideas captured during development. Review periodically and promote to
[ROADMAP.md](ROADMAP.md) or [SPRINT.md](docs/sprints/SPRINT.md) as appropriate.

- sphere_combined: outdated name
- edge radius seems to be much higher than specified 
- object specification help string is outdated
- how to use lines and emission properties in object specification?
- documentation needs update for new parchment etc material
- parchment has an ior, it shouldn't. instead it should attenuate light like... well, parchment
- ior (and maybe others) are both part of ObjectSpec and separate CLI parameter - redundant
- texture/material for plane
- Render directly to files without displaying the render window
- output to files is still flipped vertically
- check materials for real-life correctness - find references
- generalize TesseractEdgeSceneBuilder to support other 4D objects
- Add more procedural textures (wood, marble, etc.) as a texture library
- do we even support procedural textures?
- better user guide documentation
- better developer documentation
- better agent instructions for updating documentation and changelog, monitoring ci pipelines after 
  push, using glab (what else is missing?)
- films ("parchment") surface/material types
- sponge with xyz -> RGB mapping procedural texture
- better 4D rotation with mouse dragging
- Scala wrapper for libGDX to delegate var and null usage to that layer
- shadows with transparent objects
- 4D and 3D sponge cutaways
- 4D and 3D objects in a scene together
- 4D camera distance with shift mouse wheel
- examples for mixed-metallic (0 < metallic < 1) materials
- fractional sponges in OptiX
- parametrized surfaces in 3D and 4D
- multiple planes, as well as zero
- Rounded edges on cubes/sponges
