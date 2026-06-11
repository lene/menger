# TODO

Quick notes and ideas. Promote to ROADMAP.md or a sprint plan when ready to schedule.

## Unscheduled

- sbt-updates plugin (com.timushev.sbt:sbt-updates) incompatible with sbt 1.12.6 — NPE on load. Replaced by Renovate (Sprint 28.8).
- investigate hs_err*.log files
- investigate pending/ignored tests
- is the scala version used verbatim instead of as variable anywhere?
- contract similar to Polytope4DContract for Polyhedra
- Guidance for generating good and interesting scenes and animations (user guide)
- scaling objects in all dimensions
- shearing
- More 3- and higher dimensional objects:
  - surface of rotation (sor)
  - regular star 4-polytopes
  - Semiregular polyhedra and polytopes
  - gaussian splats
  - 4D spacetime trace of a person (or any object)
  - Parametric 2- and 3-surfaces in 4-space (see Sprint 30)
    - functions C->C 
  - 3-sphere (but how to visualize?)
    - Hopf fibration https://en.wikipedia.org/wiki/Hopf_fibration
    - complex-valued functions
  - julia sets over C
  
    - 2. A 3D generalization of the torus: in this case, one can imagine gluing together the opposite faces of a cube.

      A half-twist torus: same as #2, but one pair of surfaces is twisted by 180 degrees, like a Möbius strip.

      A quarter-twist torus: same as #2, but a pair of surfaces is joined by twisting them by 90 degrees.

      A third-twist prism: instead of looking at the faces of a cube, one can also use a six-sided prism. Here, opposite faces are also glued together, but one face is rotated by 120 degrees.

      A sixth-twist prism: same as #5, but one side is rotated by 60 degrees.

      A shape called a Hantzsche-Wendt manifold that consists of two cubes stacked on top of each other, with the faces of the cubes joined together in a complex way.

      A space consisting of infinitely many flat planes that can be twisted relative to each other.

      A space consisting of an infinitely tall “chimney”: four surfaces arranged as the sides of a parallelogram. Opposite surfaces are glued together.

      Same as #9, but one of the pairs of surfaces is rotated by 180 degrees.



- movie with steadily increasing level with 360 degree background
- MaterialX (.mtlx) support — Layers 1-3 (~3 sprints) — note: the metallic/AO/height
  map slots it needs land with PBR texture sets in Sprint 33:
  - **Layer 1 — File parsing**: Add MaterialX C++ SDK (github.com/AcademySoftwareFoundation/MaterialX)
    as CMake dep in `optix-jni/CMakeLists.txt`. Write `MtlxLoader.cpp` that uses
    `MaterialX::readFromFile()`, walks the document tree, and extracts Standard Surface /
    OpenPBR Surface parameter values (base_color, roughness, metallic, specular_ior, emission,
    opacity, normal/roughness/albedo image node filenames). Add JNI binding + Scala
    `MtlxMaterial` case class mirroring these fields.
  - **Layer 2 — Standard Surface → Material mapping**: Map extracted params to existing
    `InstanceMaterial` fields in `OptiXData.h` (base_color→color, specular_roughness→roughness,
    metallic→metallic, specular_ior→IOR, emission→emission). Wire through
    `MaterialExtractor.scala` and scene builders so `--object type=sphere:mtlx=path/to/mat.mtlx`
    loads the material. Already-supported fields need no shader changes.
  - **Layer 3 — Image node resolution**: MTLX `<image filename="...">` nodes need search-path
    resolution. Extend `TextureManager.scala` to accept an `MtlxMaterial` image list alongside
    the existing `ObjectSpec.texture` path. Resolve relative paths against a configurable
    `--mtlx-texture-dir`. Upload via existing `uploadTextureFromFile` pipeline; assign resulting
    indices to normal/roughness/albedo texture slots in `InstanceMaterial`.
  - Out of scope for this milestone: coat, sheen, subsurface, anisotropy, node graph
    evaluation, color space management. Unsupported inputs: warn + ignore.
- capture background by reading the desktop below the window, and render objects on top of that

## Scheduled (see ROADMAP.md)

Items moved to sprint plans:

- optix-jni decoupling + publication → Sprint 24/25
- Release & QA hardening → Sprint 28
- OptiX denoiser + curves primitive → Sprint 29
- Motion blur + OptiX API audit + optix-jni 1.0 prep → Sprint 30
- L-systems in 3D and 4D → Sprint 31
- Spectral dispersion → Sprint 32
- PBR texture sets → Sprint 33
- Production-quality caustics (incl. dispersive) → Sprint 34
- Depth of field → Sprint 35 (pushed back 2026-06-10)
- Wireframe rendering → Sprint 35 (pushed back 2026-06-10)
- Color by intensity / colormaps → Sprint 36 (pushed back 2026-06-10)
- Scalar/vector fields (functions) → Sprint 36 (pushed back 2026-06-10)
- 4D parametric surfaces → Sprint 37 (pushed back 2026-06-10)
- Parametric surface specializations → Sprint 37 (pushed back 2026-06-10)
- Schläfli polytope generator → Sprint 38 (pushed back 2026-06-10)
- Sponge cutaways (clipping planes) → Sprint 38 (pushed back 2026-06-10)
- Fractal subdivision on polychora → Sprint 38 (pushed back 2026-06-10)
- Video backgrounds → Sprint 27
- Repository split → Sprint 26
- Scalar and vector fields (datasets) → backlog
- Depth cue/Fog → Sprint 21.7 (done)
- higher max trace depth → Sprint 19.9 (done)
- fractional levels with IAS sponges → Sprint 19.10 (done)
