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

### 8.1.6 Fog / Depth Cue

Exponential distance attenuation applied to each ray hit in `helpers.cu`:

```
fog_factor = exp(-fog_density × t)
pixel_color = pixel_color × fog_factor + fog_color × (1 - fog_factor)
```

| Parameter | Type | Meaning |
|-----------|------|---------|
| `fog_density` | `float` (Params) | Attenuation rate; 0 disables fog (no-op branch) |
| `fog_r/g/b` | `float` (Params) | Fog blend color |

DSL: `Fog(density: Float, color: Color)`. When `density <= 0` the shader
skips the fog calculation entirely (zero overhead).

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
| Architecture | `src/test/scala/menger/ArchitectureSpec.scala` | ArchUnit | Layer dependency enforcement |
| OptiX Unit | `optix-jni/src/test/scala/` | ScalaTest | Renderer API |
| C++ Unit | `optix-jni/src/main/native/tests/` | Google Test | OptiX context |
| Visual | Manual | - | Render comparison |

All ArchUnit rules in `ArchitectureSpec` and `ArchitecturePhase2Spec` are active
(no `@Ignore` annotations). Rules enforce the layered dependency graph from §5.5.

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

## 8.9 Menger Sponge Face Subdivision

### Algorithm Overview

A Menger sponge surface is generated by recursive subdivision of faces. Each face
subdivides into 12 sub-faces per level:

- **8 unrotated sub-faces**: lie in the same plane as the parent, forming a 3x3 grid
  with the center removed (the tunnel opening)
- **4 rotated sub-faces**: perpendicular to the parent, forming the inner walls of the
  tunnel that passes through the center hole

```
 Unrotated sub-faces (parent face plane):    Rotated sub-faces (tunnel walls):

  ┌───┬───┬───┐                               ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │                               │   │ T │   │
  ├───┼───┼───┤                               ├───┼───┼───┤
  │ 4 │   │ 5 │  ← center removed            │ L │   │ R │  ← 4 tunnel walls
  ├───┼───┼───┤                               ├───┼───┼───┤
  │ 6 │ 7 │ 8 │                               │   │ B │   │
  └───┴───┴───┘                               └───┴───┴───┘
```

At each level, the face scale reduces by 1/3 and the pattern repeats recursively.

### Tunnel Wall Normal Convention

Each rotated sub-face is created by rotating the parent face 90 degrees around one of
four cardinal axes (`fold1`, `fold2`, `-fold1`, `-fold2`). The resulting normal must
point **into the tunnel** (away from solid material), because:

1. The normal determines how `runningCoordinatesShifted` positions the sub-face in 3D
   space. Specifically, the shift along the normal axis uses `normal.sign` as a
   multiplier.
2. At deeper recursion levels, this shift determines whether sub-tunnel walls are placed
   inside the solid material (correct) or inside the parent tunnel (incorrect).

### The Negative-Normal Correction

For faces whose normal has a **positive sign** (+X, +Y, +Z), the rotation axes
`(fold1, fold2, -fold1, -fold2)` naturally produce tunnel wall normals that point into
the tunnel.

For faces whose normal has a **negative sign** (-X, -Y, -Z), the same rotation axes
produce tunnel wall normals that point **out of** the tunnel (into the solid). This is
because `rotate90(axis)` and `rotate90(-axis)` produce opposite normals, and the
relationship between fold axes and tunnel geometry is sign-dependent.

The fix is to negate all rotation axes for negative-normal faces:

```scala
val axes =
  if normal.sign > 0 then Seq(f1, f2, -f1, -f2)
  else Seq(-f1, -f2, f1, f2)
```

### Concrete Example: Why This Matters at Level 2

Consider a level-1 sponge. The +Z and -Z faces each create tunnel walls along the Y
axis:

| Parent face | Tunnel wall normal | `normal.sign` | Level-2 Y shift direction |
|-------------|-------------------|---------------|--------------------------|
| +Z | Y (sign=+1) | +1 | y - delta (away from tunnel center) |
| -Z (uncorrected) | negY (sign=-1) | -1 | y + delta (toward tunnel center) |
| -Z (corrected) | Y (sign=+1) | +1 | y - delta (away from tunnel center) |

With the uncorrected -Z face, the level-2 sub-tunnel walls shift *toward* the center
of the level-1 tunnel, placing visible geometry inside the tunnel where it does not
belong ("tunnel intrusion"). The corrected version shifts them the same direction as
the +Z case.

Numerical example with `yCen = -1/6`, `shift = -1/18`:

```
Uncorrected: y = -1/6 - (-1/18) = -1/6 + 1/18 = -1/9  (inside tunnel)
Corrected:   y = -1/6 - (1/18)  = -1/6 - 1/18 = -2/9  (outside tunnel)
```

The tunnel boundary is at `|y| = 1/6`. The value `-1/9 ≈ -0.111` is closer to zero
than `-1/6 ≈ -0.167`, confirming it intrudes into the tunnel. The value
`-2/9 ≈ -0.222` is farther from zero, placing geometry safely in the solid.

### Regression Testing

`TunnelIntrusionDiagnostic` verifies this invariant: at level 2, no face center may
lie inside any of the three axis-aligned tunnels. The test checks all faces of a
`SpongeBySurface(2)` against the tunnel boundaries `|coordinate| < 1/6` for each
pair of non-normal axes.

## 8.10 Static Analysis

Three layers of static analysis run in CI:

### Scala: Scalafix + WartRemover

See §8.6. Runs on every push via `.git_hooks/pre-push` and CI `Scalafix` job.

### Scala: ArchUnit (Architecture Tests)

`ArchitectureSpec` and `ArchitecturePhase2Spec` enforce the layered dependency graph
(§5.5) at the bytecode level. All rules active — see §8.7.

### C++/CUDA: cppcheck

```yaml
# .gitlab-ci.yml — StaticAnalysis:Cppcheck job
cppcheck --enable=all --error-exitcode=1 \
  --suppressions-list=.cppcheck-suppress \
  optix-jni/src/main/native/
```

Suppressions tracked in `.cppcheck-suppress` (false positives from CUDA macros).

### C++/CUDA: clang-tidy

```yaml
# .gitlab-ci.yml — StaticAnalysis:ClangTidy job
clang-tidy -p "${BUILD_DIR}" --config-file=.clang-tidy \
  optix-jni/src/main/native/**/*.cpp
```

Rules configured in `.clang-tidy`. Requires `compile_commands.json` from a prior
CMake build step.

## 8.11 Two-Library Native Loading (Sprint 25)

`menger-geometry` introduces a second native library (`libmengergeometry.so`)
that depends on symbols from `liboptixjni.so`. Because shared library lookup
is name-based and both libraries are bundled inside JARs, the loading sequence
must be managed explicitly.

**Load order:**

1. `OptiXRenderer` companion object loads `liboptixjni.so` at class-load time
   (via `System.loadLibrary` or JAR extraction to a temp file).
2. `MengerRenderer` companion object loads `libmengergeometry.so`.
3. A `__attribute__((constructor))` function in `libmengergeometry.so` walks
   `dl_iterate_phdr` to find the already-loaded `liboptixjni.so` and re-opens
   it with `RTLD_GLOBAL`, promoting its symbols to the global symbol table.
   This allows lazy binding in `libmengergeometry.so` to resolve
   `OptiXWrapper::*` symbols without a link-time `DT_NEEDED` dependency.

**Why not `DT_NEEDED`:** `liboptixjni.so` is not on `LD_LIBRARY_PATH` when the
JVM starts; its path is only known at runtime after JAR extraction. The
`RTLD_GLOBAL` promotion approach avoids hardcoding paths.

**PTX resources:** `libmengergeometry.so` bundles `optix_shaders_menger.ptx`
as a Java classpath resource (`/native/x86_64-linux/optix_shaders_menger.ptx`).
`MengerRenderer` extracts this to `target/native/x86_64-linux/bin/` at load time
so the OptiX pipeline can find it.
