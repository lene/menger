# Menger - A Scala 3 implementation of the Menger sponge


## Current status
Very preliminary code skeleton.

### Usage

Compile code with `sbt compile`, test with `sbt test`, run with `sbt run`, and `sbt console`
for a Scala 3 REPL.

## To Do (very rough roadmap)
- Implement a square surface in 3D
- Implement a cube in 3D
- CI for tests, linting, SAST, and artifact generation and deployment
- Replace the square surfaces of the cube with the generator for the menger sponge
- Implement a function that generates a menger sponge of a given depth
- Repeat for a 4D cube and a 4D menger sponge
- Implement 4D transformations and 4D to 3D projections in CUDA
- Repeat for even higher dimensionalities
- Implement a Raytracer in CUDA/Optix
- Implement 3D/4D/ND mazes
- Implement a 3D/4D/ND maze solver
- Abstract the graphics routines to a generic 3D rendering library in Scala3