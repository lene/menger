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

### 8.1.4 Shadow Rays

Cast shadow rays from hit points to light sources:

```
shadow_origin = hit_point + normal × ε
shadow_direction = normalize(light_position - hit_point)

If ray hits geometry → shadow (ambient only)
If ray misses → fully lit
```

## 8.2 Alpha Channel Convention

**CRITICAL: Standard Graphics Alpha (never confuse this)**

| Alpha Value | Meaning |
|-------------|---------|
| α = 0.0 | Fully transparent (no opacity, no absorption) |
| α = 1.0 | Fully opaque (full opacity, maximum absorption) |

Applies to: OptiX shaders, Beer-Lambert absorption, Scala Color objects, all tests.

## 8.3 Color Handling

### Unified Color API

```scala
// menger.common.Color
case class Color(r: Float, g: Float, b: Float, a: Float = 1.0f)

object Color:
  def fromRGB(r: Float, g: Float, b: Float): Color
  def fromRGBA(r: Float, g: Float, b: Float, a: Float): Color
  def fromHex(hex: String): Color  // "#RRGGBB" or "#RRGGBBAA"

  val WHITE = Color(1f, 1f, 1f, 1f)
  val BLACK = Color(0f, 0f, 0f, 1f)
```

### Hex Format

```
#RRGGBB   → RGB, alpha = 1.0 (opaque)
#RRGGBBAA → RGBA, alpha from AA byte
```

## 8.4 Error Handling

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

## 8.5 Functional Style Enforcement

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

## 8.6 Testing Strategy

### Test Categories

| Category | Location | Framework | Purpose |
|----------|----------|-----------|---------|
| Scala Unit | `src/test/scala/` | ScalaTest | Logic, parsing |
| OptiX Unit | `optix-jni/src/test/scala/` | ScalaTest | Renderer API |
| C++ Unit | `optix-jni/src/test/native/` | Google Test | OptiX context |
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

## 8.7 Performance Patterns

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
