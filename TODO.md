# TODO

Quick notes and ideas. Promote to ROADMAP.md or a sprint plan when ready to schedule.

## Unscheduled

- add render time per frame and per ray to stats (also: what does the "primary" ray stat mean, 
  exactly? looks like it's always 0)
- fractional levels with IAS sponges
- higher max trace depth
- rot for x/y/z rotation
- seems like metallic materials are transparent? they should not
- publish OptiX JNI as a separate project - should cover the full OptiX API, not just the ray 
  tracing pipeline.
- Library layer for other Java/Scala clients to use OptiX JNI without Menger's scene graph or 
  rendering pipeline.
- Guidance for generating good and interesting scenes and animations (user guide)
- More 3- and higher dimensional objects:
  - construction methods listed in https://hi.gher.space/wiki/Shape
  - Regular polytopes
  - regular star 4-polytopes.
  - Semiregular polyhedra and polytopes
  - construction from Schläfli symbols (algorithmic generator for `{p,q}` / `{p,q,r}`)
  - gaussian splats
  - 4D spacetime trace of a person (or any object)
- IAS-driven infinite level of detail for 3D sponge: shipped in Sprint 18.4 as 
  `--objects type=sponge-recursive-ias:level=N` (integer 1..14, capped by OptiX 
  MAX_TRAVERSABLE_GRAPH_DEPTH).
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
    ```cpp
    // Pseudo-code for OptiX Intersection Program
    extern "C" __global__ void __intersection__menger4d() {
      float4 ray_origin = get_ray_origin_4d(); 
      float4 ray_dir = get_ray_dir_4d();
      
      // The "Fold" Loop (The Generator in Reverse)
      for (int i = 0; i < MAX_ITERATIONS; i++) {
          // 1. ABS: Fold space into the positive quadrant (Symmetry)
          pos = abs(pos);
          
          // 2. ROTATE: Apply the specific 4D rotations of your generator
          if (pos.x - pos.y < 0) swap(pos.x, pos.y); // Fold diagonals
          
          // 3. SCALE: Scale up (The Menger expansion)
          pos = pos * 3.0;
          
          // 4. SHIFT: Move origin to the new sub-cell center
          pos = pos - offset_vector;
          
          // 5. HOLE CHECK: Are we inside the empty "cross"?
          if ( is_in_empty_void(pos) ) {
              // Ray passed through a hole - no hit
              return; 
          }
      }
      
      // If we survived the loop, we hit the solid fractal surface
      reportIntersection(t_hit, 0);
    }
    ```

- movie with steadily increasing level with 360 degree background
  - movies as textures instead of png
- PBR Textures
- capture background by reading the desktop below the window, and render objects on top of that

## Scheduled (see ROADMAP.md)

Items moved to sprint plans:

- Wireframe rendering → backlog (stylistic, OptiX edge geometry)
- Multiple planes → Sprint 19 (planes as first-class geometry)
- DSL syntax for all render settings → Sprint 17.4
- Scalar and vector fields → backlog (functions first, datasets later)
- Depth cue/Fog → backlog
- Parametric surface specializations → backlog
- Color by intensity → backlog (general, including volumes)
