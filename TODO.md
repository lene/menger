# TODO

## Near-term (current priorities)
- ensure PTX are in the right place
  > Packaged app can't find optixjni library
  > The packaged application failure is expected because it's trying to load the
  > native library from the system path rather than the bundled one. This is a known
  > packaging issue but doesn't affect the correctness of the code.
- encapsulate lighting, caustics and aa etc. options to keep down number of parameters
- disable on-the-fly change of image dimension with resizing of menger main window
- print cli help on errors

## Roadmap (Sprints 5-11)

See [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md) for full details.

### Sprint 5: Triangle Mesh Foundation
- Add `OptixBuildInputTriangleArray` support
- JNI interface for vertex/index buffers
- Basic triangle closest-hit shader
- Proof-of-concept cube rendering

### Sprint 6: Cube Primitive
- Scala Cube → OptiX vertex export
- Per-face normals and materials
- CLI: `--object cube`

### Sprint 7: Multiple Objects
- Scene graph / object list
- Per-object transforms
- Multiple `--object` flags

### Sprint 8: Sponge Mesh Export
- Export `Seq[Face]` to triangle buffer
- Handle large face counts
- **🎯 v0.5 Milestone: Full Mesh Support**

### Sprint 9-10: Animation
- Keyframe animation system
- Object transform interpolation
- Frame sequence rendering to PNG
- Easing functions, multi-object animation

### Sprint 11: Scene Description Language
- Declarative scene files (YAML/JSON)
- Object, material, light definitions
- Animation keyframes in scene file

## Backlog (future)
- caustics (deferred - algorithm issues, branch `feature/caustics` preserved)
- composites
- render coordinate cross
- more primitives (cylinders, cones, torus)
- 4D sponge in OptiX
- real-time preview mode
- GPU instancing
