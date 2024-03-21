# Menger - A Scala 3 implementation of the Menger sponge in three and higher dimensions

A [Menger Sponge](https://en.wikipedia.org/wiki/Menger_sponge) is a fractal generated by subdividing
a cube into 27 smaller cubes of a third of the side length and removing the center cube and the 
cubes that share a face with it, leaving a hole in the center of each cube face. For each step of 
the generation of the fractal, this process is repeated for each of the remaining cubes, in theory 
repeating ad infinitum, but in practice it is stopped after a small number of steps due to 
processing time and memory constraints. 

The Menger Sponge is a 3D analog of the one-dimensional 
[Cantor Set](https://en.wikipedia.org/wiki/Cantor_set) and the two-dimensional 
[Sierpinski Carpet](https://en.wikipedia.org/wiki/Sierpinski_carpet). Its 
[Hausdorff dimension](https://en.wikipedia.org/wiki/Hausdorff_dimension) is 
$\log_3 20 = \frac{\log{20}}{\log{3}} \approx 2.727$.

## Alternative generation process

Instead of replacing a cube with 20 smaller cubes at every step of the generation process, the same
result can be achieved by subdividing each square face into 9 smaller squares, removing the center 
square and adding 4 squares that define the hole punched by removing the cube in the middle of the 
face. This process results in 12 smaller squares, arranged around the hole in the face. Repeating 
this process for every step of the generation process leads to the Menger Sponge fractal.

In contrast to generating a Menger Sponge from cubes, this version of the sponge has no internal 
surfaces, only the outer surface of the sponge. This process results in a computational complexity 
of $O(12^n)$ for the number of squares generated at each step, as opposed to $O(20^n)$ for the cube 
subdivision process.

## Higher-dimensional analogs

The four dimensional analog of a cube, the [Tesseract](https://en.wikipedia.org/wiki/Tesseract), can 
be treated the same way as the cube to generate the four dimensional Menger Sponge analog. The 
Tesseract is subdivided into 81 smaller Tesseracts, leaving out all Tesseracts that border on the 
center of a face of the original Tesseract for a total of 48 remaining Tesseracts.

The 4D sponge's Hausdorff dimension is $\log_3 48 = \frac{\log{48}}{\log{3}} \approx 3.524$.

It is equally possible to generate a four-dimensional Menger Sponge analog by subdividing each face
and replacing the center square with the faces of the hole in the center of the face, leading to 16
squares for each original square face. This process results in a computational complexity of 
$O(16^n)$ instead of $O(48^n)$ for the Tesseract subdivision process.


# Usage

Compile code with `sbt compile`, test with `sbt test`, run with `sbt run`, and `sbt console`
for a Scala 3 REPL.

## Options
- `--timeout <float>`
- `--sponge-type <cube|face|tesseract|tesseract-sponge>`
- `--projection-screen-w <float>`
- `--projection-eye-w <float>`
- `--rot-x-w <float>`
- `--rot-y-w <float>`
- `--rot-z-w <float>`
- `--level <int>`
- `--lines`
- `--width <int>`
- `--height <int>`
- `--antialias-samples <int>`


# Status
## Done
- Implement a square surface in 3D
- Implement a cube in 3D
- CI for tests, linting, SAST, and artifact generation and deployment
- Replace the square surfaces of the cube with the generator for the menger sponge
- Implement a function that generates a menger sponge of a given depth
- Repeat for a 4D cube and a 4D menger sponge

## Doing
- Generate a 4D menger sponge by subdividing a Tesseract's face into 16 smaller faces

## To Do (brain dump, very rough roadmap)
- Implement 4D transformations and 4D to 3D projections in CUDA
- Repeat for even higher dimensionalities
- Implement a Raytracer in CUDA/Optix
- Implement 3D/4D/ND mazes
- Implement a 3D/4D/ND maze solver
- Abstract the graphics routines to a generic 3D rendering library in Scala3
