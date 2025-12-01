# TODO

Quick notes and ideas. For detailed planning see [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md).

## Done

- ~~fix ctrl-q~~ - Refactored input controllers with shared BaseKeyController, OptiXInputMultiplexer
- ~~disable window resize~~ - Window resize disabled for OptiX mode (v0.3)

## Backlog

- MAX_INSTANCES = 64 is going to lead to problems i suppose
- move menger to a separate subproject so that all projects are on the same level
- split build.sbt, one for every subproject
- analyze and improve test coverage
- caustics improvements (deferred - algorithm issues, branch `feature/caustics` preserved)
- composites
- render coordinate cross
- more primitives (cylinders, cones, torus)
- 4D sponge in OptiX
- GPU instancing of multiple instances of the same primitive
