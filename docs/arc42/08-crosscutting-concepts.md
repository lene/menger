# 8. Crosscutting Concepts

## 8.1 Rendering Physics

### 8.1.1 Fresnel Reflectance

Surface reflectance via **Schlick approximation** of Fresnel equations:

```
R₀ = ((n₁ - n₂) / (n₁ + n₂))²
R(θ) = R₀ + (1 - R₀)(1 - cos θ)⁵
```

| Material | IOR | R₀ (perpendicular) |
|----------|-----|-------------------|
| Air | 1.0 | - |
| Water | 1.33 | 2% |
| Glass | 1.5 | 4% |
| Diamond | 2.42 | 17% |

**Behavior:** Reflectance increases from R₀ at perpendicular incidence to 100% at grazing angles.

### 8.1.2 Snell's Law (Refraction)

```
n₁ sin(θ₁) = n₂ sin(θ₂)

η = n₁ / n₂
k = 1 - η² (1 - cos²θ₁)

If k < 0: Total internal reflection
Else: Refracted direction = η·I + (η·cosθ₁ - √k)·N
```

### 8.1.3 Beer-Lambert Law (Absorption)

Exponential intensity decrease through absorbing medium:

```
I(d) = I₀ · exp(-α · d)
```

| Variable | Meaning |
|----------|---------|
| I₀ | Initial intensity |
| d | Distance through medium |
| α | Absorption coefficient |

**Color interpretation:**
- **Alpha (a):** Opacity/absorption strength (0=transparent, 1=opaque)
- **RGB:** Wavelength-dependent absorption (color tint)

```
α_r = -log(color.r) × color.a × scale
α_g = -log(color.g) × color.a × scale
α_b = -log(color.b) × color.a × scale
```

### 8.1.4 Area Lights and Soft Shadows

Area lights model extended light sources (disks) that produce soft shadow penumbras by
casting multiple shadow rays per shading point and averaging the results.

```
For each shadow sample i ∈ [1..N]:
    sample_point = area_light_center + random_disk_point(radius)
    shadow_ray_i → sample_point
    visibility_i = 1.0 if unoccluded, 0.0 if blocked

soft_shadow_factor = sum(visibility_i) / N
```

| Parameter | Meaning |
|-----------|---------|
| `position` | Center of the disk light |
| `normal` | Orientation of the disk (emission direction) |
| `radius` | Disk radius in scene units |
| `shadowSamples` | Number of shadow rays per shading point (1–16) |

**Design choice:** `shadowSamples` is specified per light in the `area:` CLI format rather than
as a global `--shadow-samples` flag. This allows different area lights in the same scene to use
different sampling densities (e.g., a large soft key light at 16 samples, a small fill at 4).

When `shadowSamples = 1`, area lights behave identically to point lights but with an
extended emission position (no soft shadow). The penumbra width scales with `radius / distance`.

### 8.1.5 Shadow Rays

Cast shadow rays from hit points to light sources:

```
shadow_origin = hit_point + normal × ε
shadow_direction = normalize(light_position - hit_point)

If ray hits geometry → shadow (ambient only)
If ray misses → fully lit
```

#### Colored Transparent Shadows (Phase 1)

When `--transparent-shadows` is enabled, shadow rays carry RGB attenuation instead of scalar
alpha. The closesthit shadow program computes per-channel attenuation from the hit object's
material:

```
attenuation_rgb = alpha × (1 - material_color_rgb)
```

| Channel | Effect |
|---------|--------|
| attenuation = 0 | No shadow (fully transparent) |
| attenuation = 1 | Full shadow (fully opaque) |
| High R, low G/B | Shadow blocks red, passes green/blue (cyan tint) |

The lighting shader applies shadow attenuation per channel to each light contribution:

```
contribution_c = light_contribution_c × (1 - attenuation_c)    for c ∈ {R, G, B}
```

When the flag is off, all three channels carry the same scalar alpha value (backward-compatible).

**Limitation (Phase 1):** Only the closest hit object contributes shadow color. Overlapping
transparent objects require anyhit accumulation (Phase 2, see TD-6).

## 8.2 Alpha Channel Convention

**CRITICAL: Standard Graphics Alpha (never confuse this)**

| Alpha Value | Meaning |
|-------------|---------|
| α = 0.0 | Fully transparent (no opacity, no absorption) |
| α = 1.0 | Fully opaque (full opacity, maximum absorption) |

Applies to: OptiX shaders, Beer-Lambert absorption, Scala Color objects, all tests.

## 8.3 Parametric Surface Tessellation

Parametric surfaces are defined by a continuous mapping `f : (u, v) → ℝ³` over a rectangular
parameter domain `[u_min, u_max] × [v_min, v_max]`.

```
f(u, v) = Vec3(x(u,v), y(u,v), z(u,v))
```

`ParametricTessellator` discretises this function into a triangle mesh:

1. **Sample grid:** Evaluate `f` at each `(u_i, v_j)` of an `(uSteps+1) × (vSteps+1)` grid.
2. **Build quads:** For each interior grid cell, form two triangles from the four corner samples.
3. **Seam welding:** When `closedU = true`, the last column wraps to column 0. Same for `closedV`.
4. **Normal estimation:** Per-vertex normals are computed via cross product of adjacent edge vectors
   using central differences where interior vertices exist, forward/backward at boundaries.

| Parameter | Effect |
|-----------|--------|
| `uSteps` / `vSteps` | Mesh resolution; higher = smoother, more triangles |
| `closedU` | Connects last u column to first (cylinder, torus, Klein bottle) |
| `closedV` | Connects last v row to first (torus, Klein bottle) |

**Example functions (built-in scenes):**

| Scene | `f(u, v)` |
|-------|-----------|
| `parametric-sphere` | `r(sin u cos v, cos u, sin u sin v)` |
| `parametric-torus` | `((R + r cos v) cos u, (R + r cos v) sin u, r sin v)` |
| `parametric-klein-bottle` | Non-orientable; self-intersecting at seam |
| `parametric-moebius` | Half-twist strip; `closedU`, `closedV=false` |

The resulting mesh is treated as a standard `TriangleMesh` scene object and passes through the
same BVH build and OptiX hit-group pipeline as any other triangle geometry.

## 8.4 Color Handling

### Unified Color API

```scala
// menger.common.Color
case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f)

object Color:
  def fromRGB(r: Int, g: Int, b: Int): Color      // 0-255 range
  def fromRGBA(r: Int, g: Int, b: Int, a: Int): Color
  def fromHex(hex: String): Color  // "#RRGGBB" or "#RRGGBBAA"

  val LIGHT_GRAY: Color = Color(200/255f, 200/255f, 200/255f)
```

### Hex Format

```
#RRGGBB   → RGB, alpha = 1.0 (opaque)
#RRGGBBAA → RGBA, alpha from AA byte
```

## 8.5 Error Handling

### Scala Layer

No exceptions. Use `Try`, `Either`, or `Option`:

```scala
def loadLibrary(): Try[Unit] = Try {
  System.loadLibrary("optix_jni")
}

def parseColor(hex: String): Either[String, Color] =
  if hex.matches("#[0-9A-Fa-f]{6,8}") then Right(...)
  else Left(s"Invalid color: $hex")
```

### C++/CUDA Layer

Macro-based error checking:

```cpp
#define CUDA_CHECK(call) \
  do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
      throw std::runtime_error(cudaGetErrorString(result)); \
    } \
  } while(0)

#define OPTIX_CHECK(call) \
  // Similar pattern for OptixResult
```

## 8.6 Functional Style Enforcement

### Wartremover Rules

```scala
// build.sbt
Warts.Var,           // No mutable variables
Warts.While,         // No while loops
Warts.AsInstanceOf,  // No unsafe casts
Warts.Throw          // No exceptions
```

### Scalafix Rules

```
OrganizeImports      // Import ordering
DisableSyntax.noNulls    // No null literals
DisableSyntax.noReturns  // No return statements
```

## 8.7 Testing Strategy

### Test Categories

| Category | Location | Framework | Purpose |
|----------|----------|-----------|---------|
| Scala Unit | `src/test/scala/` | ScalaTest | Logic, parsing |
| OptiX Unit | `optix-jni/src/test/scala/` | ScalaTest | Renderer API |
| C++ Unit | `optix-jni/src/main/native/tests/` | Google Test | OptiX context |
| Visual | Manual | - | Render comparison |

### Test Patterns

```scala
class FooTest extends AnyFlatSpec with Matchers:
  "Foo" should "do something" in:
    val result = foo()
    result shouldBe expected

  it should "handle edge case" in:
    // ...
```

## 8.8 Performance Patterns

### GPU Optimization

1. **Scene data in Params (not SBT)**
   - Reduces SBT rebuilds
   - Faster parameter updates

2. **Lazy geometry evaluation**
   ```scala
   lazy val faces: Seq[Face] = generateFaces()
   lazy val mesh: Model = buildMesh(faces)
   ```

3. **Sponge mesh caching**
   - Cache per level
   - Reuse for multiple instances

### Memory Management

- RAII in C++ (automatic cleanup)
- Explicit `cudaFree` for GPU buffers
- `dispose()` pattern in Scala for native resources
