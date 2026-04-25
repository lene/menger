# TODO

Quick notes and ideas. Promote to ROADMAP.md or a sprint plan when ready to schedule.

## Unscheduled

- publish OptiX JNI as a separate project - should cover the full OptiX API, not just the ray tracing pipeline.
- Library layer for other Java/Scala clients to use OptiX JNI without Menger's scene graph or rendering pipeline.
- Guidance for generating good and interesting scenes and animations (user guide)
- Diagnostic log messages for failed (all-red) renderings and cancel the render with a failure
- More 3- and higher dimensional objects:
  - construction methods listed in https://hi.gher.space/wiki/Shape
  - Regular polytopes
  - regular star 4-polytopes.
  - Semiregular polyhedra and polytopes
  - construction from Schläfli symbols (algorithmic generator for `{p,q}` / `{p,q,r}`)
  - gaussian splats
  - 4D spacetime trace of a person (or any object)
- IAS-driven infinite level of detail for 3D sponge:
  The "Holy Grail" for OptiX: Recursive Instancing
  If you are using OptiX for a Menger Sponge, you shouldn't be generating millions of vertices at 
  all. You should use Hardware Instancing.Because the Menger Sponge is perfectly self-similar, you 
  can exploit the Instance Acceleration Structure (IAS) to reduce your memory usage from Gigabytes 
  to Kilobytes.
  How it works: Instead of unwrapping the geometry in your script, you unwrap the scenegraph in 
  OptiX. 
  - Level 1 (The Atom): Create a single Geometry Acceleration Structure (GAS) containing just 
    the 20 squares of the generator. Upload this to VRAM once. (Size: Tiny).
  - Level 2 (The Node): Create an Instance Acceleration Structure (IAS). It contains 20 Instances. 
    Each instance points to the Level 1 GAS. Each instance has a transformation matrix 
    (Scale $1/3$, Translate to corner).
  - Level 3: Create an IAS containing 20 instances that point to the Level 2 IAS. 
  - Level $N$: Create an IAS containing 20 instances pointing to Level $N-1$. 
  The Math: Traditional Method (Level 6): Store $20^6 = 64,000,000$ squares. (~3 GB VRAM).
  Instancing Method (Level 6): Store 6 levels $\times$ 20 matrices. (~10 KB VRAM).
  Result:You can render a Level 15 Menger Sponge on a standard GPU. The limit is no longer VRAM; it 
  is the floating point precision of the matrices and the traversal stack depth (which you can 
  configure in OptiX).
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
  - Why this beats everything else:
    This is how modern fractal renderers (like **Marble Marcher** or **Mandelbulb3D**) work.
    1.  **Infinite Resolution:** You are limited only by the `float` precision, not vertex RAM. You 
        can zoom in until the universe ends.
    2.  **4D Native:** Because the math happens in your CUDA code (not the RT core's fixed-function 
        hardware), you can use 5x5 matrices, quaternions, or any 4D logic you want.
    3.  **Zero Memory:** Your VRAM usage is essentially 0MB.

- movie with steadily increasing level with 360 degree background
  - movies as textures instead of png
- PBR Textures

## Scheduled (see ROADMAP.md)

Items moved to sprint plans:

- Wireframe rendering → backlog (stylistic, OptiX edge geometry)
- Multiple planes → Sprint 19 (planes as first-class geometry)
- DSL syntax for all render settings → Sprint 17.4
- Scalar and vector fields → backlog (functions first, datasets later)
- Depth cue/Fog → backlog
- Parametric surface specializations → backlog
- Color by intensity → backlog (general, including volumes)
