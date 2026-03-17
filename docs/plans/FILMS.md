# Implementation Plan: Film/Parchment Materials and Cylinder Edge Rendering

**Feature Summary:** Add new semi-transparent "film" and "parchment" material presets with emission support, and implement true cylinder geometry for rendering edges in OptiX ray tracing mode.

---

## Requirements Specification

### 1. New Material Presets

**Film Material:**
- Default color: White (neutral, user-configurable)
- High transparency: alpha = 0.2 (80% transparent)
- Smooth/glossy: roughness = 0.1
- Low IOR: 1.1 (minimal refraction)
- Emission: 0.0 (default, configurable)
- CLI: `material=film:color=#00FFFF:emission=2.0`

**Parchment Material:**
- Default color: Beige/tan (#F5DEB3 = rgb 245,222,179)
- Medium transparency: alpha = 0.4 (60% transparent)
- Slightly rough: roughness = 0.4
- Low IOR: 1.2
- Emission: 0.0 (default, configurable)
- CLI: `material=parchment:roughness=0.6`

### 2. Emission Property

- Add `emission: Float` to Material class
- Default: 0.0 (no emission)
- Range: 0.0-10.0 (soft limit, not enforced)
- Affects rendering: emissive surfaces glow and contribute light
- CLI: `emission=VALUE`

### 3. Cylinder Primitive for Edges

- New cylinder geometry type in OptiX
- Defined by: start point, end point, radius
- Material support: fully textured with emission
- Radius configurable via CLI: `edgeRadius=VALUE`
- Used for tesseract edges and other 4D objects

### 4. Edge Material Control

- Separate material specification for edges in overlay-style rendering
- CLI: `edgeMaterial=NAME` or `edgeColor=#RRGGBB:edgeEmission=VALUE`
- Edges can be emissive (glowing) independent of face materials

---

## Architecture Overview

### Current Material System
```
CLI (ObjectSpec) → Material extraction → OptiXRenderer JNI → C++ Wrapper → GPU Structs → CUDA Shaders
```

### New Components
```
1. Material.scala: film/parchment presets + emission property
2. OptiXRenderer: addCylinderInstance() method
3. Native: Cylinder IAS geometry + intersection
4. CUDA: Cylinder ray intersection shader
5. ObjectSpec: edgeRadius, edgeMaterial CLI parameters
```

---

## Implementation Phases

### Phase 1: Add Emission Property to Material System

*This phase adds the foundational emission capability to all materials.*

#### 1.1 Update Material Case Class

**File:** `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/Material.scala`

- Add `emission: Float = 0.0f` parameter after `specular`
- Add `def withEmissionOpt(e: Option[Float]): Material = e.fold(this)(v => copy(emission = v))`
- Update all presets (Glass, Water, Diamond, Chrome, Gold, Copper) to explicitly set `emission = 0.0f`
- Update factory methods (matte, plastic, metal, glass) with `emission = 0.0f` default

#### 1.2 Update CLI Parsing

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/ObjectSpec.scala`

- Add `emission=VALUE` to documentation (lines 76-93)
- Update `parseMaterialOverrides` to parse emission using `parseOptionalFloat`
- Add `.withEmissionOpt(emission)` to material override chain

#### 1.3 Update JNI Boundary

**File:** `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`

- Add `emission: Float` parameter to `addSphereInstanceNative` and `addTriangleMeshInstanceNative`
- Update wrapper methods `addSphereInstance` and `addTriangleMeshInstance` to pass `material.emission`

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/JNIBindings.cpp`

- Add `jfloat emission` to native method signatures
- Forward emission to C++ wrapper calls

#### 1.4 Update C++ Instance Management

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXWrapper.h`

- Add `float emission = 0.0f` to `addSphereInstance` and `addTriangleMeshInstance` signatures

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/OptiXWrapper.cpp`

- Add `float emission;` to ObjectInstance struct
- Set `inst.emission = emission;` in both add methods
- Copy `mat.emission = inst.emission;` in buildIAS

#### 1.5 Update GPU Data Structures

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXData.h`

- Add `float emission;` to InstanceMaterial struct (after specular)
- Reduce padding from `padding[2]` to `padding[1]` to maintain 48-byte alignment
- Add `float emission;` to MaterialProperties struct
- Adjust padding to maintain 64-byte alignment

#### 1.6 Update CUDA Shaders (if accessible)

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/shaders/sphere_combined.cu`

- Read `material.emission` in closest-hit programs
- Add emissive contribution to final color: `float3 emissive = make_float3(material.emission * color.r, material.emission * color.g, material.emission * color.b);`
- Combine with lighting: `final_color = emissive + lighting_calculation();`

**Note:** If shader source is not accessible, document required shader changes for later implementation.

---

### Phase 2: Add Film and Parchment Presets

*This phase creates the new material presets using the emission property.*

#### 2.1 Add Film Preset

**File:** `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/Material.scala`

Add after existing presets (around line 73):

```scala
val Film = Material(
  color = White.copy(a = 0.2f),  // 20% opaque (80% transparent)
  ior = 1.1f,
  roughness = 0.1f,
  metallic = 0.0f,
  specular = 0.5f,
  emission = 0.0f
)
```

#### 2.2 Add Parchment Preset

Add after Film:

```scala
val Parchment = Material(
  color = Color(245f/255f, 222f/255f, 179f/255f, 0.4f),  // Beige/tan, 40% opaque
  ior = 1.2f,
  roughness = 0.4f,
  metallic = 0.0f,
  specular = 0.3f,
  emission = 0.0f
)
```

#### 2.3 Update fromName Lookup

**File:** `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/Material.scala`

Add to `fromName` method (around line 91):

```scala
case "film" => Film
case "parchment" => Parchment
```

#### 2.4 Add Tests

**File:** `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/MaterialPresetSuite.scala`

Add tests for film and parchment presets:
- Verify color (including alpha channel)
- Verify ior, roughness, metallic, specular, emission values
- Test fromName lookup (case-insensitive)

---

### Phase 3: Add Cylinder Primitive to OptiX

*This phase implements ray-traced cylinder geometry for edge rendering.*

#### 3.1 Design Cylinder Representation

**Cylinder Definition:**
- Start point: `float3 p0`
- End point: `float3 p1`
- Radius: `float r`
- Axis: `float3 axis = normalize(p1 - p0)`
- Length: `float length = distance(p1, p0)`

**Ray-Cylinder Intersection:**
- Infinite cylinder intersection (quadratic equation)
- Cap intersection (disk tests at p0 and p1)
- Return closest intersection within [p0, p1] range

#### 3.2 Add Cylinder IAS Geometry (C++)

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXData.h`

Add cylinder geometry type:

```cpp
enum GeometryType {
    SPHERE = 0,
    TRIANGLE_MESH = 1,
    CYLINDER = 2  // NEW
};

struct CylinderData {
    float3 p0;       // Start point (12 bytes)
    float radius;    // Radius (4 bytes)
    float3 p1;       // End point (12 bytes)
    float padding;   // Alignment (4 bytes)
    // Total: 32 bytes
};
```

Update InstanceMaterial to store geometry_type = CYLINDER.

#### 3.3 Add Cylinder Instance Management (C++)

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXWrapper.h`

Add method:

```cpp
std::optional<int> addCylinderInstance(
    float p0_x, float p0_y, float p0_z,
    float p1_x, float p1_y, float p1_z,
    float radius,
    float r, float g, float b, float a,
    float ior = 1.0f,
    float roughness = 0.5f,
    float metallic = 0.0f,
    float specular = 0.5f,
    float emission = 0.0f
);
```

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/OptiXWrapper.cpp`

Implement addCylinderInstance:
1. Create CylinderData struct
2. Upload to GPU buffer
3. Create custom primitive IAS
4. Create ObjectInstance with CYLINDER geometry type
5. Store material parameters

#### 3.4 Add Cylinder JNI Bindings

**File:** `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`

Add Scala wrapper:

```scala
@native def addCylinderInstanceNative(
    p0_x: Float, p0_y: Float, p0_z: Float,
    p1_x: Float, p1_y: Float, p1_z: Float,
    radius: Float,
    r: Float, g: Float, b: Float, a: Float,
    ior: Float, roughness: Float, metallic: Float, specular: Float, emission: Float
): Int

def addCylinderInstance(
    p0: Vec3[Float], p1: Vec3[Float], radius: Float, material: Material
): Option[Int] =
    val id = addCylinderInstanceNative(
        p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, radius,
        material.color.r, material.color.g, material.color.b, material.color.a,
        material.ior, material.roughness, material.metallic, material.specular, material.emission
    )
    if id >= 0 then Some(id) else None
```

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/JNIBindings.cpp`

Add JNI native method that forwards to OptiXWrapper::addCylinderInstance.

#### 3.5 Implement Cylinder Ray Intersection (CUDA)

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/shaders/cylinder.cu` (NEW)

Create new CUDA file with:

1. **Intersection program:**
   ```cuda
   extern "C" __global__ void __intersection__cylinder() {
       const CylinderData* cylinder = ...; // Get from SBT
       const float3 ray_orig = optixGetWorldRayOrigin();
       const float3 ray_dir = optixGetWorldRayDirection();

       // Ray-cylinder intersection math
       // 1. Infinite cylinder intersection (quadratic equation)
       // 2. Check if intersection is within [p0, p1] range
       // 3. Cap intersections (disk tests)
       // 4. optixReportIntersection() for closest hit
   }
   ```

2. **Closest-hit program:**
   - Calculate surface normal at intersection point
   - Use existing material shading (same as sphere/mesh)
   - Support emission parameter

3. **Any-hit program:**
   - Transparency support (alpha channel)

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/include/PipelineManager.h`

Add cylinder programs to pipeline configuration.

**File:** `/home/lene/workspace/menger/optix-jni/src/main/native/PipelineManager.cpp`

- Load cylinder.cu PTX module
- Create cylinder program groups (intersection, closest-hit, any-hit)
- Add to SBT records

---

### Phase 4: Add Edge Extraction and Rendering for Tesseract

*This phase enables cylinder-based edge rendering for 4D objects.*

#### 4.1 Extract Tesseract Edges

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/objects/higher_d/Tesseract.scala`

The edges are already computed (line 43):

```scala
lazy val edges: Seq[(Vector[4], Vector[4])] = ...
```

These need to be projected to 3D and converted to cylinders.

#### 4.2 Create TesseractWithEdges Scene Builder

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/engines/scene/TesseractEdgeSceneBuilder.scala` (NEW)

Create new SceneBuilder:

```scala
class TesseractEdgeSceneBuilder(
    renderer: OptiXRendererWrapper,
    edgeRadius: Float,
    faceMaterial: Material,
    edgeMaterial: Material
) extends SceneBuilder {

  def build(specs: List[ObjectSpec]): Try[Unit] = {
    specs.foreach { spec =>
      // 1. Extract tesseract from spec
      // 2. Get projection parameters
      // 3. Rotate and project faces to 3D → triangle meshes (existing)
      // 4. Rotate and project edges to 3D → cylinders (NEW)
      // 5. Add triangle mesh instances for faces
      // 6. Add cylinder instances for edges
    }
  }
}
```

Key logic:
- For each edge `(v0_4d, v1_4d)`:
  1. Apply 4D rotation
  2. Project to 3D: `p0_3d = projection(rotated_v0)`, `p1_3d = projection(rotated_v1)`
  3. Call `renderer.addCylinderInstance(p0_3d, p1_3d, edgeRadius, edgeMaterial)`

#### 4.3 Integrate with OptiX Engine

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/engines/OptiXEngine.scala`

Update scene building logic:
- Check if tesseract has `edgeMaterial` or `edgeRadius` parameters
- If so, use TesseractEdgeSceneBuilder instead of TriangleMeshSceneBuilder
- Otherwise, use existing triangle mesh builder (faces only)

---

### Phase 5: Add CLI Parameters

*This phase exposes edge rendering controls via command-line interface.*

#### 5.1 Add Edge Parameters to ObjectSpec

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/ObjectSpec.scala`

Add new optional fields to ObjectSpec case class:

```scala
case class ObjectSpec(
  // ... existing fields ...
  edgeRadius: Option[Float] = None,
  edgeMaterial: Option[Material] = None,
  edgeColor: Option[Color] = None,
  edgeEmission: Option[Float] = None
)
```

#### 5.2 Parse Edge Parameters

Update `ObjectSpec.parse` to handle:
- `edgeRadius=VALUE` (float, default 0.02)
- `edgeMaterial=NAME` (preset name like "film")
- `edgeColor=#RRGGBB` (overrides edge material color)
- `edgeEmission=VALUE` (overrides edge material emission)

**Parsing logic:**

```scala
private def parseEdgeMaterial(
    kvPairs: Map[String, String],
    edgeColor: Option[Color],
    edgeEmission: Option[Float]
): Either[String, Option[Material]] =
  kvPairs.get("edgeMaterial").map { name =>
    Material.fromName(name)
      .toRight(s"Unknown edge material: $name")
      .map { mat =>
        mat.copy(
          color = edgeColor.getOrElse(mat.color),
          emission = edgeEmission.getOrElse(mat.emission)
        )
      }
  }.getOrElse {
    // If no edgeMaterial but edgeColor/edgeEmission provided, create material
    if edgeColor.isDefined || edgeEmission.isDefined then
      Right(Some(Material(
        color = edgeColor.getOrElse(Color.White),
        emission = edgeEmission.getOrElse(0.0f)
      )))
    else
      Right(None)
  }
```

#### 5.3 Update Documentation

**File:** `/home/lene/workspace/menger/menger-app/src/main/scala/menger/ObjectSpec.scala`

Add to documentation (lines 76-93):

```
Edge Rendering (OptiX only):
  edgeRadius=VALUE       Radius of cylinder edges (default: 0.02)
  edgeMaterial=PRESET    Material preset for edges (film, parchment, etc.)
  edgeColor=#RRGGBB      Edge color override
  edgeEmission=VALUE     Edge emission override (0.0-10.0)

Examples:
  type=tesseract:material=film:edgeMaterial=film:edgeEmission=3.0
  type=tesseract:material=glass:edgeColor=#00FFFF:edgeEmission=5.0:edgeRadius=0.03
```

---

### Phase 6: Testing and Verification

#### 6.1 Material Tests

**File:** `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/MaterialUnitSuite.scala`

- Test emission property (default, boundary values, withEmissionOpt)
- Test film preset (color, alpha, ior, roughness, emission)
- Test parchment preset (color, alpha, ior, roughness, emission)
- Test fromName lookup for "film" and "parchment"

**File:** `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/MaterialPresetSuite.scala`

Add rendering tests if visual test infrastructure exists.

#### 6.2 Cylinder Tests

**File:** `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/CylinderSuite.scala` (NEW)

- Test addCylinderInstance (returns valid ID)
- Test cylinder with different materials (film, parchment, emissive)
- Test cylinder radius variations
- Visual test: render single cylinder to image, verify appearance

**C++ Tests:** (if Google Test suite accessible)

- Test ray-cylinder intersection math
- Test cap intersections
- Test normal calculations

#### 6.3 Integration Tests

**File:** `/home/lene/workspace/menger/menger-app/src/test/scala/menger/ObjectSpecSuite.scala`

- Test parsing `edgeRadius=VALUE`
- Test parsing `edgeMaterial=film`
- Test parsing `edgeColor=#00FFFF`
- Test parsing `edgeEmission=3.0`
- Test combined parameters

**End-to-End Test:**

Create test case that renders tesseract with film faces and emissive cyan edges:

```bash
sbt "run --optix --objects type=tesseract:material=film:edgeMaterial=film:edgeColor=#00FFFF:edgeEmission=5.0:edgeRadius=0.03 --output tesseract-glow.png"
```

Verify:
- Faces are semi-transparent white
- Edges are glowing cyan
- Emission creates visible glow

---

## Critical Implementation Details

### 1. Alpha Channel Convention

**CRITICAL:** Follow AGENTS.md alpha convention:
- `alpha = 0.0` → **FULLY TRANSPARENT** (no opacity)
- `alpha = 1.0` → **FULLY OPAQUE** (full opacity)

Film uses `alpha = 0.2` (20% opaque, 80% transparent).

### 2. Cylinder-Ray Intersection Math

The ray-cylinder intersection uses the quadratic formula:

**Given:**
- Ray: `R(t) = O + t*D` (origin O, direction D)
- Cylinder axis: `A = normalize(p1 - p0)`
- Cylinder center line: `C(s) = p0 + s*A`

**Intersection equation:**
```
|R(t) - C(s)|² = r²
```

Solving yields quadratic: `a*t² + b*t + c = 0`

**Discriminant:** `Δ = b² - 4ac`
- `Δ < 0`: no intersection
- `Δ ≥ 0`: two intersection points `t = (-b ± √Δ) / (2a)`

**Cap tests:**
- Project intersection point onto cylinder axis
- Verify `0 ≤ s ≤ length` for body hits
- Separate disk-ray intersection for caps at p0 and p1

### 3. Cylinder Normal Calculation

At intersection point `P`:

**Body normal:** `N = normalize((P - C(s)) - ((P - C(s)) · A) * A)`

Where `s` is the projection parameter.

**Cap normal:** `N = A` (for p1 cap) or `N = -A` (for p0 cap)

### 4. GPU Memory Alignment

All GPU structs must maintain proper alignment:

**CylinderData:** 32 bytes (4 floats × 8 = 32)
- float3 p0 (12) + float radius (4) + float3 p1 (12) + float padding (4) = 32 ✓

**InstanceMaterial:** 48 bytes
- With emission added, reduce padding[2] to padding[1]

### 5. Build Order Dependencies

Compilation must proceed in order:

1. Scala Material.scala changes (case class, presets)
2. Scala OptiXRenderer.scala (JNI signatures)
3. C++ JNIBindings.cpp (native methods)
4. C++ OptiXWrapper (instance management)
5. C++ OptiXData.h (GPU structs)
6. CUDA cylinder.cu (ray intersection)
7. CUDA sphere_combined.cu (emission shader logic)

**Build command:** `sbt compile` (compiles native and Scala together)

---

## Verification Checklist

### Phase 1: Emission
- [ ] Material case class has emission parameter (default 0.0f)
- [ ] All presets set emission = 0.0f explicitly
- [ ] withEmissionOpt helper works
- [ ] ObjectSpec parses emission=VALUE
- [ ] JNI signatures include emission parameter
- [ ] C++ ObjectInstance stores emission
- [ ] GPU InstanceMaterial includes emission (48-byte aligned)
- [ ] CUDA shaders read and use emission
- [ ] Tests pass (MaterialUnitSuite)

### Phase 2: Film/Parchment
- [ ] Film preset: White, alpha=0.2, ior=1.1, roughness=0.1
- [ ] Parchment preset: Beige (#F5DEB3), alpha=0.4, ior=1.2, roughness=0.4
- [ ] fromName("film") and fromName("parchment") work
- [ ] Tests pass (MaterialPresetSuite)

### Phase 3: Cylinders
- [ ] CylinderData struct defined (32 bytes)
- [ ] addCylinderInstance JNI binding works
- [ ] C++ creates custom primitive IAS for cylinders
- [ ] cylinder.cu implements ray intersection correctly
- [ ] Cylinder normals calculated correctly (body and caps)
- [ ] Cylinder SBT records created
- [ ] Single cylinder renders correctly in test

### Phase 4: Tesseract Edges
- [ ] TesseractEdgeSceneBuilder extracts and projects edges
- [ ] Edges converted to 3D cylinders with correct radius
- [ ] Edge material applied correctly
- [ ] Faces and edges render together

### Phase 5: CLI
- [ ] edgeRadius=VALUE parsed
- [ ] edgeMaterial=PRESET parsed
- [ ] edgeColor=#RRGGBB parsed
- [ ] edgeEmission=VALUE parsed
- [ ] Combined parameters work together
- [ ] Documentation updated

### Phase 6: Testing
- [ ] MaterialUnitSuite: emission tests pass
- [ ] MaterialPresetSuite: film/parchment tests pass
- [ ] CylinderSuite: cylinder rendering tests pass
- [ ] ObjectSpecSuite: edge parameter parsing tests pass
- [ ] End-to-end test: tesseract with glowing cyan edges renders correctly
- [ ] All existing tests still pass (regression check)

---

## Example Usage

### Basic Film Material
```bash
sbt "run --optix --objects type=sphere:pos=0,0,0:material=film --output sphere-film.png"
```

### Parchment Material with Custom Roughness
```bash
sbt "run --optix --objects type=cube:material=parchment:roughness=0.6 --output cube-parchment.png"
```

### Tesseract with Film Faces and Glowing Cyan Edges
```bash
sbt "run --optix --objects type=tesseract:material=film:edgeMaterial=film:edgeColor=#00FFFF:edgeEmission=5.0:edgeRadius=0.03 --output tesseract-glow.png"
```

### Emissive Sphere
```bash
sbt "run --optix --objects type=sphere:material=glass:emission=3.0:color=#FF0000 --output sphere-emissive.png"
```

---

## Estimated Implementation Effort

| Phase | Description | Estimated Time |
|-------|-------------|----------------|
| 1 | Emission property (Scala + C++ + CUDA) | 3-4 hours |
| 2 | Film/parchment presets | 1 hour |
| 3 | Cylinder primitive (C++ + CUDA) | 6-8 hours |
| 4 | Tesseract edge extraction/rendering | 3-4 hours |
| 5 | CLI parameters | 2-3 hours |
| 6 | Testing and verification | 3-4 hours |
| **Total** | | **18-24 hours** |

**Note:** Cylinder ray intersection is the most complex component (Phase 3). The math is well-established but requires careful CUDA implementation and testing.

---

## Risks and Mitigations

### Risk 1: Cylinder Intersection Performance
**Issue:** Ray-cylinder intersection is more expensive than ray-sphere (quadratic solve + cap tests).

**Mitigation:**
- Use tight bounding boxes for IAS
- Consider LOD: distant edges rendered as thin meshes instead of cylinders
- Profile performance with large tesseract sponges

### Risk 2: Shader Code Access
**Issue:** May not have access to modify CUDA shader source.

**Mitigation:**
- Complete all Scala/C++ infrastructure first
- Document shader changes needed in detail
- Emission will be stored but not rendered until shaders updated
- This is acceptable for phased rollout

### Risk 3: Emission Lighting Integration
**Issue:** Emissive surfaces should affect indirect lighting (global illumination).

**Mitigation:**
- Start with simple additive emission (glow effect)
- Full emission→lighting requires path tracing enhancements (defer to future work)
- Document as known limitation

### Risk 4: Edge Radius Scaling
**Issue:** Fixed edge radius may look wrong at different tesseract sizes or camera distances.

**Mitigation:**
- Default radius (0.02) chosen for size=1.0 tesseract
- Users can override via edgeRadius parameter
- Consider automatic scaling in future: `edgeRadius = size * 0.02`

---

## Future Enhancements (Out of Scope)

These are **not** part of this implementation but noted for future work:

1. **Automatic edge detection:** Extract edges from any triangle mesh, not just tesseracts
2. **Taper support:** Cylinders with different start/end radii (cones)
3. **Edge LOD:** Switch to thin meshes for distant/small edges
4. **Bloom/glow post-processing:** Enhance emissive appearance with bloom shader
5. **Emission affects lighting:** Make emissive surfaces act as area lights in path tracer
6. **Texture mapping on cylinders:** UV coordinates for textured edges

---

## Critical Files to Modify

### Scala (Frontend)
- `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/Material.scala` - Core material model
- `/home/lene/workspace/menger/optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` - JNI renderer interface
- `/home/lene/workspace/menger/menger-app/src/main/scala/menger/ObjectSpec.scala` - CLI parameter parsing
- `/home/lene/workspace/menger/menger-app/src/main/scala/menger/engines/scene/TesseractEdgeSceneBuilder.scala` - NEW: Edge scene builder
- `/home/lene/workspace/menger/menger-app/src/main/scala/menger/engines/OptiXEngine.scala` - Engine integration

### C++ (Native Layer)
- `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXData.h` - GPU data structures
- `/home/lene/workspace/menger/optix-jni/src/main/native/include/OptiXWrapper.h` - Wrapper interface
- `/home/lene/workspace/menger/optix-jni/src/main/native/OptiXWrapper.cpp` - Instance management
- `/home/lene/workspace/menger/optix-jni/src/main/native/JNIBindings.cpp` - JNI entry points
- `/home/lene/workspace/menger/optix-jni/src/main/native/include/PipelineManager.h` - Pipeline config
- `/home/lene/workspace/menger/optix-jni/src/main/native/PipelineManager.cpp` - Pipeline setup

### CUDA (Shaders)
- `/home/lene/workspace/menger/optix-jni/src/main/native/shaders/cylinder.cu` - NEW: Cylinder intersection
- `/home/lene/workspace/menger/optix-jni/src/main/native/shaders/sphere_combined.cu` - Emission rendering

### Tests
- `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/MaterialUnitSuite.scala` - Material tests
- `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/MaterialPresetSuite.scala` - Preset tests
- `/home/lene/workspace/menger/optix-jni/src/test/scala/menger/optix/CylinderSuite.scala` - NEW: Cylinder tests
- `/home/lene/workspace/menger/menger-app/src/test/scala/menger/ObjectSpecSuite.scala` - CLI parsing tests

---

## Dependencies and Prerequisites

### Build Environment
- CUDA Toolkit 12.0+
- NVIDIA OptiX SDK 9.0+
- CMake 3.18+
- C++17 compiler
- Java 21+
- sbt 1.11+

### Knowledge Requirements
- Ray tracing fundamentals
- OptiX API (custom primitives, SBT, intersection programs)
- CUDA programming
- JNI (Java Native Interface)
- Scala 3 (case classes, Option, Try)

### Code Review Focus
- GPU struct alignment (critical for performance)
- Ray intersection correctness (use reference implementations)
- Alpha channel convention (alpha=0 is transparent!)
- Emission shader integration (ensure emission is additive, not multiplicative)

---

## Success Criteria

Implementation is complete when:

1. ✅ Film and parchment materials render correctly with expected transparency and finish
2. ✅ Emission property works for all materials (surfaces glow when emission > 0)
3. ✅ Cylinders render correctly in OptiX with proper ray intersection and normals
4. ✅ Tesseract edges render as emissive cylinders with configurable radius
5. ✅ CLI parameters (edgeRadius, edgeMaterial, edgeColor, edgeEmission) parse and apply correctly
6. ✅ All tests pass (unit, integration, visual)
7. ✅ Example tesseract with glowing cyan edges (matching reference image) renders successfully
8. ✅ No performance regression for existing features
9. ✅ Code follows AGENTS.md standards (no var, functional style, proper error handling)
10. ✅ Documentation updated (ObjectSpec help, CHANGELOG.md)
