# Sprint 5: Triangle Mesh Foundation + Cube

**Created:** 2025-11-22
**Status:** 📋 PLANNED
**Estimated Effort:** 12-18 hours
**Branch:** TBD (create from `main` when starting)

## Overview

Establish the infrastructure for triangle mesh rendering in OptiX with a working cube primitive. This sprint lays the foundation for all future mesh-based geometry (sponges, 4D projections, arbitrary objects).

### Sprint Goal

Render an opaque and glass cube via `--object cube`, proving the triangle mesh pipeline works end-to-end.

### Success Criteria

- [ ] `--object cube` renders a solid colored cube
- [ ] `--object cube` with transparency/IOR renders a glass cube with refraction
- [ ] Triangle mesh infrastructure is reusable for future geometry types
- [ ] All new code has tests
- [ ] Existing 897+ tests still pass

---

## Architectural Decisions

These decisions affect future sprints. Document rationale to avoid regret later.

### AD-1: Triangle geometry coexists with analytical sphere

**Decision:** Keep the existing analytical sphere; add triangle mesh as a parallel path.

**Rationale:**
- Analytical sphere is optimized and working well
- Breaking change would disrupt existing functionality
- Different geometry types can coexist in same scene (Sprint 6)

**Impact:** Need to handle both sphere and triangle closest-hit shaders.

### AD-2: Build GAS for triangles, prepare for IAS in Sprint 6

**Decision:** Single Geometry Acceleration Structure (GAS) for Sprint 5. Instance Acceleration Structure (IAS) deferred to Sprint 6.

**Rationale:**
- Single cube doesn't need instancing
- IAS adds complexity (transforms, instance SBT offsets)
- Sprint 6 (multiple objects) is the right place for IAS

**Impact:** Sprint 5 cube has no independent transform; Sprint 6 will refactor to IAS.

### AD-3: Vertex format = position + normal (UVs in Sprint 7)

**Decision:** Vertex buffer contains position (float3) and normal (float3). No UVs yet.

**Rationale:**
- Position needed for geometry
- Normal needed for lighting (diffuse shading, refraction)
- UVs only needed for textures (Sprint 7: Materials)
- Adding UVs later is straightforward (extend vertex struct)

**Impact:** Vertex stride = 24 bytes (6 floats). Will change in Sprint 7.

### AD-4: Per-face normals for cube (24 vertices)

**Decision:** Use per-face normals, meaning 24 vertices (4 per face × 6 faces) not 8.

**Rationale:**
- Cube has sharp edges; per-vertex normals would smooth them incorrectly
- Per-face = duplicate vertices at corners, each with face's normal
- Future smooth objects can use per-vertex normals (same infrastructure)

**Impact:** Cube vertex count = 24, index count = 36 (12 triangles × 3).

### AD-5: 32-bit indices from the start

**Decision:** Use 32-bit (unsigned int) indices, not 16-bit.

**Rationale:**
- Sponge at level 3+ exceeds 65535 vertices (16-bit limit)
- 32-bit has negligible performance impact on modern GPUs
- Avoids breaking change later

**Impact:** Index buffer uses `unsigned int` throughout.

### AD-6: Add triangle shader to sphere_combined.cu

**Decision:** Add triangle closest-hit shader to existing `sphere_combined.cu`.

**Rationale:**
- Single PTX file simplifies build and loading
- Shader can share utility functions (refraction, Fresnel, etc.)
- Refactor to separate files in Sprint 6 if file becomes unwieldy

**Impact:** `sphere_combined.cu` grows; may need refactoring later.

### AD-7: Cube refraction uses face-based inside/outside tracking

**Decision:** Track ray inside/outside state based on face normal dot product, not analytical containment.

**Rationale:**
- Cube doesn't have analytical inside test like sphere
- `dot(ray_dir, normal) > 0` means exiting; `< 0` means entering
- Same approach works for any convex mesh

**Impact:** Refraction code differs slightly from sphere; shared Fresnel/Beer-Lambert.

---

## Step 5.1: Triangle Mesh Infrastructure (4-6 hours)

### Goal
OptiX can receive vertex/index buffers from Scala and build acceleration structure.

### Tasks

#### 5.1.1 Define vertex and index buffer structures in OptiXData.h

```cpp
struct TriangleMeshData {
    float3* vertices;        // Position data
    float3* normals;         // Per-vertex normals
    unsigned int* indices;   // Triangle indices (3 per triangle)
    unsigned int num_vertices;
    unsigned int num_triangles;
    float3 color;            // Object color (RGB, 0-1)
    float alpha;             // Transparency (0=transparent, 1=opaque)
    float ior;               // Index of refraction
};
```

#### 5.1.2 Add triangle GAS building to OptiXWrapper.cpp

- New method: `buildTriangleMeshGAS(TriangleMeshData* mesh)`
- Use `OptixBuildInputTriangleArray`
- Set vertex format to `OPTIX_VERTEX_FORMAT_FLOAT3`
- Set index format to `OPTIX_INDICES_FORMAT_UNSIGNED_INT3`
- Build GAS with `optixAccelBuild()`

#### 5.1.3 Create JNI interface for mesh data

- New JNI method: `setTriangleMesh(float[] vertices, float[] normals, int[] indices, ...)`
- Marshal arrays from Java to native buffers
- Handle memory lifecycle (allocate on set, free on dispose)

#### 5.1.4 Add triangle closest-hit program to SBT

- New closest-hit program: `__closesthit__triangle`
- Add to Shader Binding Table alongside sphere hit program
- Set up hit group for triangle geometry

### Tests

- **Unit test:** Verify vertex/index buffers are correctly passed through JNI
- **Unit test:** Verify GAS builds without errors for valid mesh data
- **Unit test:** Verify invalid mesh data (null, empty, mismatched counts) is rejected

### Caveats

- **Memory management:** GPU buffers must be freed when mesh changes or renderer disposes
- **GAS rebuild:** Changing mesh requires rebuilding GAS (expensive); cache if mesh unchanged
- **SBT complexity:** Adding hit groups increases SBT size; keep organized

---

## Step 5.2: Basic Opaque Cube Rendering (3-4 hours)

### Goal
Render a solid-colored cube using the triangle mesh infrastructure.

### Tasks

#### 5.2.1 Create CubeGeometry in Scala

New file: `optix-jni/src/main/scala/menger/optix/geometry/CubeGeometry.scala`

```scala
object CubeGeometry:
  def generate(size: Float = 1.0f): TriangleMeshData =
    // 24 vertices (4 per face × 6 faces)
    // 36 indices (2 triangles × 3 vertices × 6 faces)
    // Per-face normals pointing outward
```

#### 5.2.2 Implement triangle closest-hit shader

In `sphere_combined.cu`:

```cuda
extern "C" __global__ void __closesthit__triangle()
{
    // Get triangle data from SBT
    // Interpolate normal using barycentric coordinates (if per-vertex)
    // Or use face normal directly (if per-face, stored in SBT)

    // Basic diffuse shading
    float3 normal = /* get normal */;
    float3 hit_point = /* compute from ray */;
    float3 color = /* get from mesh data */;

    // Apply lighting (reuse existing light loop)
    float3 result = computeLighting(hit_point, normal, color);

    // Set payload
    setPayloadColor(result);
}
```

#### 5.2.3 Wire up cube rendering path

- If `--object cube` specified, create CubeGeometry and set mesh
- Build triangle GAS instead of (or alongside) sphere
- Render with triangle closest-hit program

#### 5.2.4 Handle cube in miss program (plane rendering)

- Cube casts shadows on plane (if shadows enabled)
- Shadow ray needs to test against triangle GAS

### Tests

- **Visual test:** Cube renders with correct shape (6 faces visible from angle)
- **Visual test:** Cube has correct face normals (each face shaded differently)
- **Visual test:** Cube casts shadow on plane (if shadows enabled)
- **Unit test:** CubeGeometry produces correct vertex/index counts
- **Unit test:** CubeGeometry normals point outward for each face

### Caveats

- **Cube position:** Where is the cube? Same position as sphere? Configurable?
  - Decision: Same default position as sphere (centered at origin, above plane)
  - Position/scale CLI options can come in Sprint 6

- **Face winding:** Must be consistent (counter-clockwise) for correct culling
  - OptiX uses CCW by default for front-facing

---

## Step 5.3: Glass Cube Support (3-5 hours)

### Goal
Render a transparent cube with refraction, reusing existing Fresnel/Beer-Lambert code.

### Tasks

#### 5.3.1 Extend triangle closest-hit for transparency

- Check alpha value; if < 1.0, handle as transparent
- Compute refracted ray direction using Snell's law
- Determine if entering or exiting based on `dot(ray_dir, normal)`
- Apply Fresnel reflection/refraction split
- Trace refracted ray recursively

#### 5.3.2 Implement Beer-Lambert absorption for cube

- Track distance traveled inside cube
- Apply `exp(-absorption * distance)` color attenuation
- Reuse existing Beer-Lambert code from sphere

#### 5.3.3 Handle multiple internal bounces

- Cube can have internal reflections (total internal reflection)
- Ray may exit through different face than expected
- Limit bounce depth (reuse existing MAX_TRACE_DEPTH)

#### 5.3.4 Test glass cube with various IOR values

- IOR 1.0: No refraction (pass-through)
- IOR 1.5: Glass-like refraction
- IOR 2.4: Diamond-like refraction

### Tests

- **Visual test:** Glass cube refracts background correctly
- **Visual test:** Glass cube shows caustic-like bright spots (from refraction focus)
- **Visual test:** Colored glass cube tints refracted light
- **Unit test:** Refraction direction correct for cube face normals
- **Comparison test:** Glass cube at IOR 1.0 looks same as fully transparent

### Caveats

- **Corner cases (literally):** Rays hitting cube corners/edges may have numerical issues
  - Mitigation: Small epsilon offset when spawning secondary rays

- **Performance:** Glass cube traces more rays than opaque
  - Expected: 2-4x more rays for glass vs opaque
  - Monitor with ray statistics

- **Visual difference from sphere:** Cube refraction looks different (flat faces vs curved)
  - Expected behavior; cube focuses light into lines, sphere into points

---

## Step 5.4: CLI Integration (2-3 hours)

### Goal
`--object cube` works from command line with proper validation and help.

### Tasks

#### 5.4.1 Add --object option to CLI parser

In `MengerCLIOptions.scala`:

```scala
val objectType: ScallopOption[String] = opt[String](
  required = false,
  default = Some("sphere"),
  descr = "Object to render: sphere, cube"
)

validateOpt(objectType) { obj =>
  if Set("sphere", "cube").contains(obj.toLowerCase) then Right(())
  else Left(s"Unknown object type: $obj. Valid types: sphere, cube")
}
```

#### 5.4.2 Wire CLI option to renderer

- Parse `--object` value
- Create appropriate geometry (sphere or cube)
- Configure renderer

#### 5.4.3 Ensure existing sphere options work with cube

- `--sphere-color` → should this become `--object-color`?
  - Decision: Keep `--sphere-color` for backward compatibility; add `--cube-color`
  - Unify in Sprint 7 (Materials) with proper material system

- `--sphere-alpha`, `--sphere-ior` → same pattern

#### 5.4.4 Update --help output

- List valid object types
- Show cube-specific options

### Tests

- **CLI test:** `--object cube` parses correctly
- **CLI test:** `--object invalid` shows error message
- **CLI test:** `--object sphere` still works (backward compatibility)
- **CLI test:** Cube color/alpha/IOR options work
- **Integration test:** Full render with `--object cube --shadows --antialiasing`

### Caveats

- **CLI complexity growing:** Many options now
  - This is why Scene Description Language is planned
  - Keep CLI functional but accept it's a stopgap

- **Option naming:** sphere-specific vs generic
  - Accept inconsistency for now; unify in Sprint 7

---

## Files to Create/Modify

### New Files

| File | Description |
|------|-------------|
| `optix-jni/src/main/scala/menger/optix/geometry/CubeGeometry.scala` | Cube vertex/index generation |
| `optix-jni/src/main/scala/menger/optix/geometry/TriangleMeshData.scala` | Data class for mesh buffers |
| `optix-jni/src/test/scala/menger/optix/geometry/CubeGeometryTest.scala` | Tests for cube generation |
| `optix-jni/src/test/scala/menger/optix/TriangleMeshTest.scala` | Tests for triangle rendering |

### Modified Files

| File | Changes |
|------|---------|
| `optix-jni/src/main/native/include/OptiXData.h` | Add TriangleMeshData struct |
| `optix-jni/src/main/native/OptiXWrapper.cpp` | Add buildTriangleMeshGAS(), mesh handling |
| `optix-jni/src/main/native/OptiXWrapper.h` | Declare new methods |
| `optix-jni/src/main/native/JNIBindings.cpp` | Add setTriangleMesh JNI binding |
| `optix-jni/src/main/native/shaders/sphere_combined.cu` | Add __closesthit__triangle |
| `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` | Add setTriangleMesh method |
| `src/main/scala/menger/MengerCLIOptions.scala` | Add --object option |
| `src/main/scala/menger/OptiXResources.scala` | Handle object type selection |

---

## Risk Mitigation

### Risk 1: GAS build fails for triangle input

**Likelihood:** Low (OptiX triangle support is well-documented)
**Impact:** High (blocks all triangle rendering)
**Mitigation:**
- Follow OptiX samples closely (optixTriangle sample)
- Validate input data before build
- Log detailed errors on failure

### Risk 2: Triangle shader doesn't integrate with existing lighting

**Likelihood:** Medium (different code paths)
**Impact:** Medium (cube looks different than sphere)
**Mitigation:**
- Extract lighting code into shared device functions
- Test with same light configurations as sphere
- Compare side-by-side

### Risk 3: Glass cube refraction looks wrong

**Likelihood:** Medium (complex geometry)
**Impact:** Low (can ship opaque-only first)
**Mitigation:**
- Test with simple cases first (IOR 1.0, single face)
- Compare with reference renderers
- Add debug visualization for ray paths if needed

### Risk 4: Performance regression

**Likelihood:** Low (triangles are fast on RTX)
**Impact:** Medium (affects user experience)
**Mitigation:**
- Benchmark before/after
- Profile with ray statistics
- Optimize only if needed

---

## Dependencies

### External
- OptiX SDK 9.0+ (already required)
- CUDA 12.0+ (already required)

### Internal
- Existing OptiXWrapper infrastructure
- Existing shader utilities (Fresnel, Beer-Lambert)
- Existing lighting system (multi-light support)

---

## Future Considerations

### Sprint 6 (Multiple Objects)
- Will need to refactor from single GAS to IAS + multiple GAS
- CubeGeometry should be reusable for per-cube-in-sponge rendering
- Consider: Should cube be a separate GAS or part of scene GAS?

### Sprint 7 (Materials)
- Will add UVs to vertex format (stride increases)
- Material system will replace per-object color/alpha/IOR
- CubeGeometry will need UV generation

### Sprint 8-10 (4D Support)
- 4D objects project to 3D triangles
- Same triangle infrastructure should handle projected meshes
- Ensure vertex buffer system is flexible for dynamic mesh updates

---

## Definition of Done

- [ ] All tasks completed
- [ ] All tests passing (new + existing 897+)
- [ ] Code compiles without warnings
- [ ] Code passes `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Visual verification: cube renders correctly (screenshot in PR)
- [ ] Performance verified: no significant regression
- [ ] Documentation: architectural decisions recorded

---

## References

- [OptiX Programming Guide - Triangle Input](https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures#triangle-build-input)
- [optixTriangle SDK sample](https://github.com/NVIDIA/OptiX_Apps/tree/master/apps/optixTriangle)
- Existing implementation: `sphere_combined.cu` (refraction code)
- Existing implementation: `OptiXWrapper.cpp` (GAS building for spheres)
