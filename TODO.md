# TODO

Quick notes and ideas. Promote to ROADMAP.md or a sprint plan when ready to schedule.

## Unscheduled

- investigate pending tests
- publish OptiX JNI as a separate project - should cover the full OptiX API, not just the ray 
  tracing pipeline.
- Library layer for other Java/Scala clients to use OptiX JNI without Menger's scene graph or 
  rendering pipeline.
- Guidance for generating good and interesting scenes and animations (user guide)
- scaling objects in all dimensions
- shearing
- More 3- and higher dimensional objects:
  - surface of rotation (sor)
  - regular star 4-polytopes
  - Semiregular polyhedra and polytopes
  - construction from Schläfli symbols (algorithmic generator for `{p,q}` / `{p,q,r}`)
  - gaussian splats
  - 4D spacetime trace of a person (or any object)
  - 3-sphere (but how to visualize?)
    - Hopf fibration https://en.wikipedia.org/wiki/Hopf_fibration
    - complex-valued functions
- The Real "Smart Idea": Procedural Intersection (SDFs)
  Since you cannot store the vertices (memory limit) and you cannot instance the geometry (4D 
  limit), the only robust solution for high-dimensional fractals in OptiX is Procedural Primitives.
  You don't upload any mesh data (no vertices, no indices). Instead, you upload the Logic.
  = The Setup (Custom Primitive)
    You create a single Custom Primitive (an Axis-Aligned Bounding Box) that represents the bounding
    volume of your entire 4D object in 3D space.
  = The Intersection Shader (`__intersection__`)
    This is where the magic happens. Instead of testing "Ray vs. Triangle," you write a loop that 
    mathematically checks "Ray vs. Fractal."
    Because Menger Sponges are Iterated Function Systems (IFS), you can reverse the logic. Instead 
    of generating the geometry, you "fold the space".

- movie with steadily increasing level with 360 degree background
  - movies as textures instead of png
- PBR Textures
- capture background by reading the desktop below the window, and render objects on top of that

## Scheduled (see ROADMAP.md)

Items moved to sprint plans:

- Wireframe rendering → backlog (stylistic, OptiX edge geometry)
- Multiple planes → Sprint 19.4 (planes as first-class geometry)
- Scalar and vector fields → backlog (functions first, datasets later)
- Depth cue/Fog → backlog
- Parametric surface specializations → backlog
- Color by intensity → backlog (general, including volumes)
- rot for x/y/z rotation → Sprint 19.7 (per-object 3D rotation via scene graph)
- render time per frame and per ray → Sprint 19.8
- higher max trace depth → Sprint 19.9 (spike)
- fractional levels with IAS sponges → Sprint 19.10 (spike)
