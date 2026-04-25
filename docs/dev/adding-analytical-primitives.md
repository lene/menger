# Adding an Analytical Primitive (IS Program Recipe)

Audience: Sprint 19+ implementers adding a new GPU-side analytical primitive
(cylinder, cone, torus, plane, parametric surface, …).

This document captures the **canonical pattern** for registering a custom OptiX
intersection (IS) program, derived from the existing cylinder implementation.
The pattern is stable; follow it verbatim unless there is a concrete reason to
deviate.

## Two Existing Patterns — Pick One

OptiX hands you two ways to feed per-primitive parameters into an IS program.
The codebase uses both:

1. **SBT-data pattern** — used by `__intersection__sphere`. Parameters are
   embedded directly in the SBT hit-group record. One record per primitive.
   Best when: parameters are tiny and fixed per geometry-type record (e.g.
   single-sphere scenes).

2. **Params-indirection pattern** — used by `__intersection__cylinder`. SBT
   record carries no per-instance data; the IS program reads
   `optixGetInstanceId()`, looks up the per-instance material in
   `params.instance_materials[]`, and uses `material.texture_index` as an
   index into a flat per-primitive buffer (`params.cylinder_data[]`).
   Best when: many instances of the primitive, dynamic parameters, or
   parameters that change per render without rebuilding the SBT.

**For Sprint 19 primitives (cone, torus, plane, parametric):
use the params-indirection pattern.** It scales to many instances, separates
geometry from SBT layout, and only requires `cudaMemcpy` for parameter updates.

## Recipe — Params-Indirection Pattern

Anchor file when in doubt: `optix-jni/src/main/native/shaders/hit_cylinder.cu`.

### 1. Define the primitive parameter struct

In `optix-jni/src/main/native/include/OptiXData.h`, alongside `CylinderData`:

```cpp
struct ConeData {
    float apex[3];
    float half_angle;
    float axis[3];
    float height;
};
```

Pad to 16- or 32-byte alignment; keep struct `__device__`-friendly (no
constructors, no virtual functions, plain trivially-copyable POD).

### 2. Add buffer + count to `Params`

In the same header, alongside `cylinder_data` / `num_cylinders`:

```cpp
ConeData*    cone_data;
unsigned int num_cones;
```

### 3. Write `hit_cone.cu`

Copy `hit_cylinder.cu` as the starting template. Required shader entry points:

| Entry point | Required for |
|---|---|
| `__intersection__cone` | radiance + shadow + photon hit groups |
| `__closesthit__cone` | radiance hit group |
| `__closesthit__cone_shadow` | opaque shadow path |
| `__anyhit__cone_shadow` | transparent-shadow accumulation |
| `__closesthit__photon` | already shared with sphere/cylinder via the photon module — only add if cone-specific photon storage is needed |

Inside `__intersection__cone`, follow the cylinder defensive pattern verbatim:

```cpp
const unsigned int instanceId = optixGetInstanceId();
if (instanceId >= params.num_instances) return;
if (!params.instance_materials)        return;
const InstanceMaterial& mat = params.instance_materials[instanceId];
const int cone_index = mat.texture_index;
if (cone_index < 0 ||
    cone_index >= static_cast<int>(params.num_cones)) return;
if (!params.cone_data)                 return;
const ConeData* cone = &params.cone_data[cone_index];
// … intersection math …
```

These guards are not optional — early renders without them produced silent
GPU faults that surfaced as the all-uniform frames the failed-render
diagnostic now catches (Sprint 18.6).

### 4. Include the new shader in the main module

In `optix-jni/src/main/native/shaders/optix_shaders.cu`:

```cpp
#include "hit_cone.cu"
```

The CMake build compiles the umbrella file to a single PTX; no separate
module is needed unless you have a strong reason (cylinder lives in the
same module despite the legacy `cylinder_module` member name in
`PipelineManager`).

### 5. Register the program groups

In `optix-jni/src/main/native/PipelineManager.cpp`, mirror the cylinder
hit-group registration block. Three program groups are required per
primitive:

```cpp
cone_hitgroup_prog_group = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__cone",
    module, "__intersection__cone");

cone_shadow_hitgroup_prog_group =
    optix_context.createHitgroupProgramGroupWithAH(
        module, "__closesthit__cone_shadow",
        module, "__anyhit__cone_shadow",
        module, "__intersection__cone");

photon_cone_hitgroup = optix_context.createHitgroupProgramGroup(
    module, "__closesthit__photon",
    module, "__intersection__cone");
```

Add the corresponding members to `PipelineManager.h`, destroy them in
`destroy()`, and feed them into the SBT builder where the cylinder hit
groups are wired up.

### 6. Wire the SBT and instance type

- Append a new geometry-type enum value where cylinder lives (e.g.
  `GEOMETRY_TYPE_CONE`).
- Reserve an SBT base offset (`STRIDE_RAY_TYPES * GEOMETRY_TYPE_CONE`).
- When the Scala layer adds a cone instance, set
  `InstanceMaterial.texture_index = cone_index` so the IS program can
  look up `params.cone_data[]`.

### 7. Scala-side wiring

- New `OptiXRenderer.addCone(...)` method that stages a `ConeData`
  upload and an instance with the correct geometry type.
- `SceneClassifier`: cones classify like cylinders (custom IS,
  IAS-compatible).
- `cli/converters`: `--objects type=cone:apex=...:half_angle=...` parser.
- A scene-builder mirroring the cylinder builder.

### 8. Tests and references

- Unit test: render one cone, snapshot pixel values.
- Mixed-geometry test: cone + sphere + cube in one scene (validates the
  multi-GAS IAS path resolved in 18.1).
- Integration-test entry in `scripts/integration-tests.sh`.
- Reference image checked in.

## Why no shared `PrimitiveParameters` abstraction

The existing two primitives (sphere, cylinder) use *different* parameter
patterns and the cylinder pattern was chosen as canonical only after
shipping. Unifying now would force a rewrite of the working sphere path
without a second concrete data point to validate the abstraction.

The canonical pattern is documented here as a recipe; the abstraction can
be extracted later, after the second analytical primitive lands and
confirms the design.

## Pipeline ceiling

`OptixPipelineSetStackSize` is configured for `maxTraversableGraphDepth = 2`
(IAS → GAS) in `OptiXContext.cpp`. New analytical primitives **do not
require raising this** — they live in their own GAS at the same depth as
the existing primitives. (Recursive IAS, Sprint 18.4, is the work item
that will raise it.)

## Checklist before opening a PR

- [ ] Header struct added and aligned
- [ ] `Params` extended with buffer + count
- [ ] `hit_<prim>.cu` mirrors cylinder defensive pattern
- [ ] Shader included in `optix_shaders.cu`
- [ ] Three program groups registered (radiance, shadow, photon)
- [ ] SBT base offset reserved
- [ ] Scala renderer entry point added
- [ ] Mixed-geometry integration test
- [ ] Reference image
- [ ] User-guide entry in `docs/USER_GUIDE.md` for the CLI flag
