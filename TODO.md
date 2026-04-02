# TODO

Quick notes and ideas. Promote to ROADMAP.md or a sprint plan when ready to schedule.

## Unscheduled

- publish OptiX JNI as a separate project - should cover the full OptiX API, not just the ray tracing pipeline.
- Library layer for other Java/Scala clients to use OptiX JNI without Menger's scene graph or rendering pipeline.
- Guidance for generating good and interesting scenes and animations (user guide)
- More 3- and higher dimensional objects:
  - construction methods listed in https://hi.gher.space/wiki/Shape
  - Regular polytopes
  - regular star 4-polytopes.
  - Semiregular polyhedra and polytopes
  - construction from Schläfli symbols (algorithmic generator for `{p,q}` / `{p,q,r}`)

## Scheduled (see ROADMAP.md)

Items moved to sprint plans:

- Wireframe rendering → backlog (stylistic, OptiX edge geometry)
- Multiple planes → Sprint 19 (planes as first-class geometry)
- DSL syntax for all render settings → Sprint 17.4
- Scalar and vector fields → backlog (functions first, datasets later)
- Depth cue/Fog → backlog
- Parametric surface specializations → backlog
- Color by intensity → backlog (general, including volumes)
