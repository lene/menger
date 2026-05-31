# menger-geometry

In-repo Menger-specific extension of `optix-jni`. **Not intended for external use.**

This module adds 4D geometry and caustics rendering on top of the generic
`optix-jni` GPU ray tracing library:

- 4D hit shaders: Menger sponge, Sierpiński gasket, hexadecachoron
- `Project4D` GPU-side projection from 4D to 3D
- Recursive IAS sponge construction
- `CausticsRenderer` (progressive photon mapping)
- `MengerRenderer` — Scala API aggregating the above, extends `OptiXRenderer`

## Dependency

`menger-geometry` depends on `optix-jni` (for `OptiXRenderer` and the OptiX
pipeline infrastructure) and `menger-common` (for domain primitives).

`menger-app` depends on `menger-geometry` and routes all 4D API calls through
`MengerRenderer`.

## Not for external use

This module is intentionally not published. It contains Menger-specific geometry
types (`Menger4DData`, `Sierpinski4DData`, `Hexadecachoron4DData`) that have no
meaning outside this project. External projects that want GPU ray tracing should
depend on `io.github.lene:optix-jni` directly — see `optix-jni/README.md`.
