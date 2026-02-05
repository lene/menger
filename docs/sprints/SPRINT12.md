# Sprint 12: Visual Quality & Material Enhancements

**Sprint:** 12 - Visual Quality Improvements
**Status:** Not Started
**Estimate:** 10-14 hours
**Branch:** `feature/sprint-12`
**Dependencies:** Sprint 10 (Scala DSL) - optional

---

## Goal

Improve visual quality and material realism with better plane materials, transparent shadows, material validation, and visual enhancements for cubes and sponges.

## Success Criteria

- [ ] Plane supports textures and materials (not just solid colors)
- [ ] Shadows work correctly with transparent/glass objects
- [ ] Material physical correctness validation and documentation
- [ ] Mixed-metallic material examples (0 < metallic < 1)
- [ ] Rounded edges on cubes and sponges (optional stretch goal)
- [ ] Updated USER_GUIDE.md with material best practices
- [ ] All tests pass (~20-25 new tests)

---

## Background

### Current State

| Feature | Status | Notes |
|---------|--------|-------|
| Sphere materials | ✅ Complete | Glass, chrome, gold, etc. working |
| Mesh materials | ✅ Complete | Cubes and sponges support materials |
| Plane rendering | ✅ Complete | Solid colors and checkerboard only |
| Transparent objects | ✅ Complete | Glass spheres render correctly |
| Shadows | ⚠️ Partial | Opaque objects cast shadows, transparent objects don't |
| Material presets | ✅ Complete | 15+ presets (Glass, Chrome, Gold, etc.) |
| Metallic parameter | ✅ Complete | Binary only (0.0 or 1.0) in practice |

### Gaps

1. **Plane materials:** Planes only support solid colors, no textures or materials
2. **Transparent shadows:** Glass objects should cast colored shadows, not block light completely
3. **Material validation:** No validation that materials are physically plausible
4. **Mixed metallic:** No examples of partial metallic values (e.g., 0.5)
5. **Sharp edges:** Cubes and sponges have perfectly sharp edges (unrealistic)

---

## Tasks

### Task 12.1: Plane Materials and Textures

**Estimate:** 3 hours

Enable planes to use full material system (textures, colors, materials).

#### Current Limitation

Planes only support `color` parameter:
```bash
menger --optix --plane Y,-2.0,#808080  # Solid gray plane
```

#### Proposed Enhancement

Support textures and materials:
```bash
# Textured plane
menger --optix --plane Y,-2.0 --plane-texture wood.png

# Material plane (e.g., chrome mirror floor)
menger --optix --plane Y,-2.0 --plane-material chrome

# Custom plane material
menger --optix --plane Y,-2.0 --plane-color #FFFFFF --plane-metallic 1.0 --plane-roughness 0.1
```

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Add new options:
```scala
val planeTexture: ScallopOption[String] = opt[String](
  name = "plane-texture", required = false,
  descr = "Texture file for plane (mutually exclusive with plane-material)"
)

val planeMaterial: ScallopOption[String] = opt[String](
  name = "plane-material", required = false,
  descr = "Material preset for plane (glass, chrome, etc.)"
)

val planeMetallic: ScallopOption[Float] = opt[Float](
  name = "plane-metallic", required = false, default = Some(0.0f),
  validate = x => x >= 0.0f && x <= 1.0f,
  descr = "Plane metallic parameter (0.0-1.0, default: 0.0)"
)

val planeRoughness: ScallopOption[Float] = opt[Float](
  name = "plane-roughness", required = false, default = Some(0.5f),
  validate = x => x >= 0.0f && x <= 1.0f,
  descr = "Plane roughness parameter (0.0-1.0, default: 0.5)"
)

// Validation: cannot specify both texture and material
validateOpt(planeTexture, planeMaterial) { (texture, material) =>
  if texture.isDefined && material.isDefined then
    Left("Cannot specify both --plane-texture and --plane-material")
  else
    Right(())
}
```

**`optix-jni/src/main/native/include/OptiXData.h`**

Add material fields to Plane struct:
```cpp
struct Plane {
    float3 normal;
    float distance;
    // Existing fields
    float3 color;
    float3 checkerColor;
    float checkerSize;

    // New material fields
    float metallic;
    float roughness;
    float ior;
    int textureIndex;  // -1 for no texture
};
```

**`optix-jni/src/main/native/shaders/optix_shaders.cu`**

Update plane shader to handle materials:
```cuda
// In __closesthit__plane()
const Plane& plane = params.plane;

// Sample texture if present
float3 baseColor = plane.color;
if (plane.textureIndex >= 0) {
    float2 uv = computePlaneUV(rayOrigin, rayDirection, plane);
    baseColor = sampleTexture(plane.textureIndex, uv);
}

// Apply material model (Matte vs Metallic)
if (plane.metallic > 0.5f) {
    // Metallic reflection
    handleMetallicOpaque(baseColor, plane.metallic, plane.roughness, ...);
} else {
    // Matte diffuse
    handleMatteOpaque(baseColor, plane.roughness, ...);
}
```

#### Tests to Add

```scala
class PlaneMaterialSpec extends AnyFlatSpec:
  it should "support textured plane" in:
    // Test plane with texture renders correctly

  it should "support material presets on plane" in:
    // Test --plane-material chrome creates reflective floor

  it should "reject both texture and material" in:
    // Test validation error
```

---

### Task 12.2: Transparent Shadows

**Estimate:** 3.5 hours

Enable glass objects to cast colored shadows (light passes through with attenuation).

#### Current Limitation

Glass objects completely block shadows (shadow ray returns occluded).

#### Proposed Enhancement

Shadow rays through glass objects should:
1. Continue tracing through the glass
2. Attenuate light by glass color and opacity
3. Allow multiple glass layers (stack depth limit)

#### Files to Modify

**`optix-jni/src/main/native/shaders/optix_shaders.cu`**

Add shadow transparency handling:
```cuda
// In shadow ray miss shader
extern "C" __global__ void __anyhit__shadow()
{
    const SphereData& sphere = /* current sphere */;
    const Material& mat = sphere.material;

    // If opaque, terminate shadow ray (full occlusion)
    if (mat.opacity >= 0.99f) {
        optixTerminateRay();
        return;
    }

    // For transparent materials, attenuate and continue
    float3 attenuation = mat.color * (1.0f - mat.opacity);

    // Update shadow payload (accumulated attenuation)
    ShadowPayload* prd = getPRD<ShadowPayload>();
    prd->attenuation *= attenuation;

    // If attenuation is very low, stop tracing (optimization)
    if (luminance(prd->attenuation) < 0.01f) {
        optixTerminateRay();
    }

    // Otherwise, ignore this hit and continue
    optixIgnoreIntersection();
}
```

#### Configuration

Add option to enable/disable transparent shadows:
```bash
menger --optix --transparent-shadows  # Enable (default: disabled)
```

#### Tests to Add

```scala
class TransparentShadowsSpec extends AnyFlatSpec:
  it should "cast colored shadows through glass" in:
    // Render scene with red glass sphere
    // Verify shadow has red tint

  it should "handle multiple glass layers" in:
    // Stack 3 glass spheres
    // Verify progressive attenuation
```

**Manual Verification:**
```bash
# Red glass sphere should cast red-tinted shadow
menger --optix --transparent-shadows \
  --objects type=sphere:pos=0,1,0:material=glass:color=#FF0000 \
  --plane Y,-2.0,#FFFFFF \
  --save-name transparent-shadow-test.png
```

---

### Task 12.3: Material Physical Correctness Validation

**Estimate:** 2 hours

Validate and document material parameters against real-world physics.

#### Scope

1. Review all material presets for physical plausibility
2. Add validation warnings for unrealistic combinations
3. Document reference values in USER_GUIDE.md

#### Material Properties to Validate

| Material | IOR Range | Roughness | Metallic | Notes |
|----------|-----------|-----------|----------|-------|
| Glass | 1.45-1.90 | 0.0-0.1 | 0.0 | Never metallic |
| Water | 1.33 | 0.0-0.05 | 0.0 | Low roughness |
| Diamond | 2.42 | 0.0 | 0.0 | High IOR |
| Chrome | ~1.5 | 0.0-0.2 | 1.0 | Always metallic |
| Gold | ~0.47 | 0.0-0.3 | 1.0 | Complex IOR |
| Plastic | 1.4-1.6 | 0.3-0.7 | 0.0 | Never metallic |

#### Files to Modify

**`menger-common/src/main/scala/menger/common/Material.scala`**

Add validation method:
```scala
case class Material(
  color: Color,
  ior: Float,
  metallic: Float,
  roughness: Float,
  opacity: Float,
  emission: Float
):
  def validate(): Seq[String] =
    val warnings = mutable.ArrayBuffer[String]()

    // Warning: Glass with metallic > 0
    if ior > 1.3f && metallic > 0.1f then
      warnings += s"Glass-like material (IOR=$ior) should not be metallic ($metallic)"

    // Warning: Metallic with low IOR
    if metallic > 0.9f && ior < 1.0f then
      warnings += s"Metallic material should have IOR ~1.5, got $ior"

    // Warning: Unrealistic IOR range
    if ior < 1.0f || ior > 3.0f then
      warnings += s"IOR $ior is outside typical range [1.0, 3.0]"

    warnings.toSeq
```

**CLI Warning:**

On startup, validate all materials and log warnings:
```scala
scene.materials.foreach { mat =>
  mat.validate().foreach { warning =>
    logger.warn(s"Material validation: $warning")
  }
}
```

#### Documentation

**`USER_GUIDE.md`** - Add "Material Reference" section:

```markdown
## Material Reference

### Index of Refraction (IOR)

Common materials:
- Air: 1.0 (reference)
- Water: 1.33
- Glass: 1.5-1.9 (window glass: ~1.52, flint glass: ~1.6)
- Plastic: 1.4-1.6
- Diamond: 2.42
- Metals: 0.2-3.0 (complex values, often use ~1.5 for simplicity)

### Metallic Parameter

- **0.0**: Dielectric (glass, plastic, water)
- **1.0**: Metal (chrome, gold, copper)
- **0.0-1.0**: Hybrid materials (rusty metal, painted surfaces)

### Roughness Parameter

- **0.0**: Perfect mirror/glass (sharp reflections)
- **0.5**: Slightly rough surface (blurred reflections)
- **1.0**: Very rough (matte, diffuse)

### Physical Constraints

- Metals (`metallic=1.0`) should have `ior ~1.5` and `opacity=1.0`
- Glass (`ior>1.4`) should have `metallic=0.0` and `opacity<1.0`
- Matte surfaces should have high roughness (`>0.5`)
```

#### Tests to Add

```scala
class MaterialValidationSpec extends AnyFlatSpec:
  it should "warn about glass with metallic" in:
    val mat = Material(Color.White, ior=1.5f, metallic=1.0f, ...)
    mat.validate() should contain ("should not be metallic")

  it should "accept physically plausible materials" in:
    val glass = Material.Glass
    glass.validate() shouldBe empty
```

---

### Task 12.4: Mixed-Metallic Material Examples

**Estimate:** 1.5 hours

Create example scenes demonstrating partial metallic values (0 < metallic < 1).

#### Background

Current presets only use `metallic=0.0` or `metallic=1.0`. Partial values create interesting hybrid materials (painted metal, rusty metal, etc.).

#### Examples to Create

**`examples/mixed-metallic-spheres.sh`**

```bash
#!/bin/bash
# Demonstrate partial metallic values

menger --optix \
  --objects type=sphere:pos=-3,0,0:metallic=0.0:roughness=0.3:color=#FF4444 \
  --objects type=sphere:pos=-1.5,0,0:metallic=0.25:roughness=0.3:color=#FF4444 \
  --objects type=sphere:pos=0,0,0:metallic=0.5:roughness=0.3:color=#FF4444 \
  --objects type=sphere:pos=1.5,0,0:metallic=0.75:roughness=0.3:color=#FF4444 \
  --objects type=sphere:pos=3,0,0:metallic=1.0:roughness=0.3:color=#FF4444 \
  --plane Y,-2.0,#CCCCCC \
  --save-name mixed-metallic.png \
  --timeout 5
```

#### Documentation

**`USER_GUIDE.md`** - Add "Partial Metallic Values" section:

```markdown
### Partial Metallic Values

Mixed metallic values create hybrid materials:

- **0.25**: Lightly painted metal (paint with metallic flakes)
- **0.5**: Rusty metal, oxidized copper
- **0.75**: Worn chrome, scratched metal

Example:
```bash
# Rusty metal sphere
menger --optix --objects type=sphere:metallic=0.5:roughness=0.6:color=#AA5533
```
```

#### Tests to Add

```scala
class MixedMetallicIntegrationSpec extends AnyFlatSpec:
  it should "render spheres with varying metallic values" in:
    // Render scene with metallic=0.0, 0.5, 1.0
    // Verify all render without errors
```

---

### Task 12.5: Rounded Edges on Cubes/Sponges (Stretch Goal)

**Estimate:** 4 hours

Add optional edge rounding to cube and sponge geometry for more realistic appearance.

**Note:** This is a stretch goal. If time is limited, defer to a future sprint.

#### Approach

Use chamfered/beveled edge vertices:
- Add parameter: `edge-radius=0.05` (default: 0.0 for sharp edges)
- Subdivide edges into curved sections
- Recompute normals for smooth shading

#### Files to Modify

**`menger-app/src/main/scala/menger/objects/Cube.scala`**

Add optional rounding:
```scala
def createRoundedCube(size: Float, edgeRadius: Float): TriangleMeshData =
  require(edgeRadius >= 0.0f && edgeRadius < size/2, "Invalid edge radius")

  if (edgeRadius < 0.001f) {
    // Sharp edges (existing implementation)
    createCube(size)
  } else {
    // Rounded edges (subdivide corners)
    createChamferedCube(size, edgeRadius)
  }
```

#### CLI Option

```scala
val edgeRadius: ScallopOption[Float] = opt[Float](
  name = "edge-radius", required = false, default = Some(0.0f),
  validate = _ >= 0.0f,
  descr = "Round edges on cubes/sponges (0.0=sharp, >0=rounded)"
)
```

---

### Task 12.6: Documentation and Examples

**Estimate:** 1 hour

Update USER_GUIDE.md with material best practices and new features.

#### Sections to Add

1. **Material Reference** (from Task 12.3)
2. **Partial Metallic Values** (from Task 12.4)
3. **Plane Materials** (from Task 12.1)
4. **Transparent Shadows** (from Task 12.2)
5. **Common Material Mistakes** (validation warnings)

#### Example Gallery Updates

Add to examples:
- Chrome mirror floor (plane material)
- Colored shadows through glass
- Mixed metallic progression
- Textured plane (wood, marble)

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 12.1 | Plane materials/textures | 3h | High |
| 12.2 | Transparent shadows | 3.5h | High |
| 12.3 | Material validation | 2h | Medium |
| 12.4 | Mixed-metallic examples | 1.5h | Medium |
| 12.5 | Rounded edges (stretch) | 4h | Low |
| 12.6 | Documentation | 1h | High |
| **Total** | | **11-15h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated with material reference
- [ ] Example scenes created and tested
- [ ] TODO.md updated

---

## Notes

### Implementation Order

Recommended order:

1. **Task 12.1** (Plane materials) - Foundation for realistic floors
2. **Task 12.3** (Material validation) - Prevents mistakes early
3. **Task 12.4** (Mixed-metallic examples) - Simple, high visual impact
4. **Task 12.2** (Transparent shadows) - Complex but high impact
5. **Task 12.5** (Rounded edges) - Optional, time permitting
6. **Task 12.6** (Documentation) - Last

### Testing Strategy

- **Unit tests**: Material validation logic
- **Integration tests**: Plane materials, transparent shadows
- **Visual tests**: Manual verification of shadow colors, reflections
- **Performance tests**: Ensure transparent shadows don't degrade fps

### Visual Quality Philosophy

This sprint focuses on **realism and polish**. Every feature adds subtle but important visual improvements:
- Textured floors add environmental context
- Colored shadows add realism to glass objects
- Validated materials prevent "uncanny valley" renders
- Rounded edges soften harsh geometry

---

## References

- TODO.md: Transparent shadows, plane materials, mixed-metallic examples
- Material physics: PBRT book, Chapter 8 (Materials)
- OptiX shadow rays: [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#ray_types)
