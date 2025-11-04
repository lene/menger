# Glass Rendering with NVIDIA OptiX - Complete Research Findings

## Executive Summary

This document contains comprehensive research findings on rendering glass and dielectric materials using NVIDIA OptiX ray tracing. Based on analysis of OptiX SDK 9.0 examples, official documentation, developer forums, and practical testing, this guide covers both fundamental concepts and advanced techniques for physically accurate glass rendering.

**Critical Finding**: OptiX built-in sphere primitives do NOT detect intersections from rays originating inside the sphere. This fundamental limitation requires using custom intersection programs for proper glass rendering.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [OptiX-Specific Limitations](#2-optix-specific-limitations)
3. [Custom Intersection Programs](#3-custom-intersection-programs)
4. [Glass Material Implementation](#4-glass-material-implementation)
5. [Physics of Glass Rendering](#5-physics-of-glass-rendering)
6. [Advanced Glass Effects](#6-advanced-glass-effects)
7. [Performance Considerations](#7-performance-considerations)
8. [Implementation Patterns](#8-implementation-patterns)
9. [Common Problems and Solutions](#9-common-problems-and-solutions)
10. [Future Enhancements](#10-future-enhancements)

---

## 1. Core Concepts

### Glass as a Dielectric Material

Glass is a **dielectric material** characterized by:
- **Transparency**: Allows light to pass through
- **Refraction**: Bends light at boundaries due to IOR (Index of Refraction)
- **Reflection**: Partial reflection based on Fresnel equations
- **Absorption**: Wavelength-dependent attenuation (colored glass)
- **Dispersion**: Wavelength-dependent IOR (prisms, rainbows)

### Ray-Glass Interactions

When a ray hits glass, four things can happen:
1. **Reflection**: Ray bounces off surface
2. **Refraction**: Ray enters glass and bends
3. **Total Internal Reflection**: Ray cannot exit due to critical angle
4. **Absorption**: Ray intensity decreases while traveling through glass

### Required Ray Information

For proper glass rendering, track:
- **Entry/Exit State**: Is ray entering or leaving the glass?
- **Surface Normal**: Must point toward incident ray
- **Distance Traveled**: For volume absorption calculations
- **Current IOR**: For nested dielectrics
- **Ray Importance**: For terminating low-contribution rays

---

## 2. OptiX-Specific Limitations

### Built-in Sphere Primitive Limitations

**Critical Discovery**: OptiX built-in sphere primitives (`OPTIX_BUILD_INPUT_TYPE_SPHERES`):
- ✅ Detect rays entering from outside
- ❌ Do NOT detect rays exiting from inside
- ❌ Cannot distinguish entry vs exit hits
- ❌ Unsuitable for glass rendering

**Evidence**: Test results show `hit_t = -1.0` for all rays originating inside spheres.

### Why This Matters

Glass rendering requires:
1. Ray enters sphere → Refracts inward
2. Ray travels through glass → Absorption applied
3. **Ray must detect exit point** → Second refraction
4. Ray continues to next object

Without step 3, glass rendering fails.

### OptiX SDK Solution

The official OptiX samples (e.g., optixWhitted) **never use built-in sphere primitives for glass**. Instead, they use:
- Custom intersection programs
- Sphere shell geometry
- Explicit entry/exit detection

---

## 3. Custom Intersection Programs

### Basic Custom Sphere Intersection

```cuda
extern "C" __global__ void __intersection__sphere()
{
    // Get sphere data from SBT
    const float3 center = /* sphere center */;
    const float radius = /* sphere radius */;

    // Ray data
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    // Ray-sphere intersection
    float3 O = ray_orig - center;
    float b = dot(O, ray_dir);
    float c = dot(O, O) - radius * radius;
    float discriminant = b * b - c;

    if (discriminant > 0.0f) {
        float sqrt_disc = sqrtf(discriminant);

        // Near intersection (entry or exit from inside)
        float t1 = -b - sqrt_disc;
        bool hit1_valid = (t1 >= ray_tmin && t1 <= ray_tmax);

        // Far intersection (exit or entry from outside)
        float t2 = -b + sqrt_disc;
        bool hit2_valid = (t2 >= ray_tmin && t2 <= ray_tmax);

        // Report BOTH intersections
        if (hit1_valid) {
            float3 hit_point = ray_orig + t1 * ray_dir;
            float3 normal = normalize((hit_point - center) / radius);
            unsigned int hit_kind = 0; // Entry
            optixReportIntersection(t1, hit_kind,
                __float_as_uint(normal.x),
                __float_as_uint(normal.y),
                __float_as_uint(normal.z));
        }

        if (hit2_valid) {
            float3 hit_point = ray_orig + t2 * ray_dir;
            float3 normal = normalize((hit_point - center) / radius);
            unsigned int hit_kind = 1; // Exit
            optixReportIntersection(t2, hit_kind,
                __float_as_uint(normal.x),
                __float_as_uint(normal.y),
                __float_as_uint(normal.z));
        }
    }
}
```

### Sphere Shell Intersection (OptiX SDK Method)

```cuda
struct SphereShell {
    float3 center;
    float radius1;  // Inner radius
    float radius2;  // Outer radius
};

extern "C" __global__ void __intersection__sphere_shell()
{
    // Complex intersection handling for hollow sphere
    // Supports 4 hit types:
    // - HIT_OUTSIDE_FROM_OUTSIDE (enter outer sphere)
    // - HIT_OUTSIDE_FROM_INSIDE (exit outer sphere)
    // - HIT_INSIDE_FROM_OUTSIDE (enter inner cavity)
    // - HIT_INSIDE_FROM_INSIDE (exit inner cavity)

    // Check if ray starts outside or inside
    float O_dot_O = dot(O, O);
    if (O_dot_O > sqr_radius2) {
        // Ray starts outside
        // Handle entry into outer sphere
    } else if (O_dot_O > sqr_radius1) {
        // Ray starts between shells
        // Can hit either inner or outer sphere
    } else {
        // Ray starts in inner cavity
        // Handle exit from inner sphere
    }
}
```

### Custom Intersection for Polygon Meshes

For triangle-based glass objects:

```cuda
// No custom intersection needed - use built-in triangle support
// OptiX handles triangles natively with hardware acceleration

// However, need to track face orientation
extern "C" __global__ void __closesthit__glass_mesh()
{
    // Get triangle vertices
    const float3* vertices = /* vertex buffer */;
    const uint3 indices = optixGetTriangleIndices();

    // Compute normal (may need to flip based on winding)
    float3 v0 = vertices[indices.x];
    float3 v1 = vertices[indices.y];
    float3 v2 = vertices[indices.z];
    float3 normal = normalize(cross(v1 - v0, v2 - v0));

    // Ensure normal points toward ray
    if (dot(normal, ray_dir) > 0) {
        normal = -normal;
    }
}
```

---

## 4. Glass Material Implementation

### Material Data Structure

```cuda
struct Glass {
    // Importance sampling
    float importance_cutoff;      // Stop tracing below this threshold
    float3 cutoff_color;          // Color when below cutoff

    // Fresnel parameters (Schlick approximation)
    float fresnel_exponent;       // Power exponent (typically 5.0)
    float fresnel_minimum;        // Minimum reflectance (R0)
    float fresnel_maximum;        // Maximum reflectance (typically 1.0)

    // Refraction
    float refraction_index;       // IOR (e.g., 1.5 for glass)
    float3 refraction_color;      // Tint for refracted rays
    int refraction_maxdepth;      // Max refraction bounces

    // Reflection
    float3 reflection_color;      // Tint for reflected rays
    int reflection_maxdepth;      // Max reflection bounces

    // Volume absorption (Beer-Lambert)
    float3 extinction_constant;   // Absorption per unit distance

    // Shadows
    float3 shadow_attenuation;    // How much light passes through
};
```

### Closest Hit Shader

```cuda
extern "C" __global__ void __closesthit__glass_radiance()
{
    // 1. Determine entry/exit
    const bool entering = (dot(ray_dir, geometric_normal) < 0);
    float3 normal = entering ? geometric_normal : -geometric_normal;

    // 2. Calculate hit points with offset
    float3 front_hit = hit_point + scene_epsilon * normal;
    float3 back_hit = hit_point - scene_epsilon * normal;

    // 3. Compute Fresnel reflectance
    float cos_theta = abs(dot(ray_dir, normal));
    float fresnel = fresnel_schlick(cos_theta,
        glass.fresnel_exponent,
        glass.fresnel_minimum,
        glass.fresnel_maximum);

    // 4. Handle refraction
    float3 refract_dir;
    if (refract(refract_dir, ray_dir, normal, glass.refraction_index)) {
        // Trace refracted ray from back hit point
        float3 refract_color = traceRadianceRay(back_hit, refract_dir);
        result += (1 - fresnel) * glass.refraction_color * refract_color;
    } else {
        // Total internal reflection
        fresnel = 1.0f;
    }

    // 5. Handle reflection
    float3 reflect_dir = reflect(ray_dir, normal);
    float3 reflect_color = traceRadianceRay(front_hit, reflect_dir);
    result += fresnel * glass.reflection_color * reflect_color;

    // 6. Apply volume absorption (if exiting)
    if (!entering && saved_entry_distance >= 0) {
        float distance = ray_t - saved_entry_distance;
        float3 absorption = exp(-glass.extinction_constant * distance);
        result *= absorption;
    }

    // 7. Update payload for next ray
    if (entering) {
        payload.entry_distance = ray_t;
        payload.current_ior = glass.refraction_index;
    } else {
        payload.entry_distance = -1;
        payload.current_ior = 1.0f;
    }
}
```

---

## 5. Physics of Glass Rendering

### Snell's Law (Refraction)

**Physical Law**: `n₁ sin θ₁ = n₂ sin θ₂`

**Vector Implementation**:
```cuda
bool refract(float3& t, const float3& i, const float3& n, const float ior)
{
    float eta = entering ? ior : 1.0f / ior;
    float cos_theta_i = -dot(i, n);
    float sin2_theta_t = eta * eta * (1 - cos_theta_i * cos_theta_i);

    if (sin2_theta_t > 1.0f) {
        // Total internal reflection
        return false;
    }

    float cos_theta_t = sqrt(1 - sin2_theta_t);
    t = eta * i + (eta * cos_theta_i - cos_theta_t) * n;
    return true;
}
```

**Critical Angle**: `θc = arcsin(n₂/n₁)`
- Glass to air: ~41.8° for IOR=1.5
- Diamond to air: ~24.4° for IOR=2.42

### Fresnel Equations

**Schlick Approximation**:
```cuda
float fresnel_schlick(float cos_theta, float exponent = 5.0f,
                      float minimum = 0.0f, float maximum = 1.0f)
{
    float r0 = (ior - 1) / (ior + 1);
    r0 = r0 * r0;
    float complement = 1.0f - cos_theta;
    return minimum + (maximum - minimum) * r0 +
           (1 - r0) * pow(complement, exponent);
}
```

**Typical Values**:
- Glass (IOR=1.5): R₀ = 0.04 (4% at normal incidence)
- Water (IOR=1.33): R₀ = 0.02 (2% at normal incidence)
- Diamond (IOR=2.42): R₀ = 0.17 (17% at normal incidence)

### Beer-Lambert Law (Volume Absorption)

**Physical Law**: `I = I₀ · e^(-α·d)`

**Implementation**:
```cuda
// Define absorption coefficients
float3 compute_extinction(float3 glass_color, float density) {
    // Darker colors = more absorption
    return -log(glass_color) * density;
}

// Apply absorption
float3 apply_absorption(float3 color, float distance) {
    return color * exp(-extinction_constant * distance);
}
```

**Color Examples**:
- Clear glass: `extinction = {0.01, 0.01, 0.01}`
- Green glass: `extinction = {0.5, 0.1, 0.5}`
- Blue glass: `extinction = {0.5, 0.5, 0.1}`

### Dispersion (Wavelength-Dependent IOR)

```cuda
// Cauchy's equation for dispersion
float ior_wavelength(float wavelength_nm) {
    float A = 1.5;  // Base IOR
    float B = 0.01; // Dispersion strength
    float lambda2 = wavelength_nm * wavelength_nm;
    return A + B / lambda2;
}

// Render with spectral sampling
float3 render_dispersive_glass() {
    float3 result = {0, 0, 0};

    // Sample red wavelength (700nm)
    float ior_r = ior_wavelength(700);
    result.r = trace_with_ior(ior_r);

    // Sample green wavelength (550nm)
    float ior_g = ior_wavelength(550);
    result.g = trace_with_ior(ior_g);

    // Sample blue wavelength (450nm)
    float ior_b = ior_wavelength(450);
    result.b = trace_with_ior(ior_b);

    return result;
}
```

---

## 6. Advanced Glass Effects

### Shadow Rays Through Glass

Glass objects cast **colored, attenuated shadows**:

```cuda
extern "C" __global__ void __anyhit__glass_occlusion()
{
    // Get shadow payload
    PayloadShadow& payload = getPayloadShadow();

    // Calculate Fresnel for shadow ray
    float3 normal = optixGetWorldNormal();
    float3 ray_dir = optixGetWorldRayDirection();
    float cos_theta = abs(dot(normal, ray_dir));

    // Attenuate shadow based on:
    // 1. Fresnel reflection (blocks some light)
    // 2. Glass color (tints shadow)
    // 3. Thickness (if available)

    float fresnel = fresnel_schlick(cos_theta, 5.0f, 0.04f, 1.0f);
    payload.attenuation *= (1 - fresnel) * glass.shadow_attenuation;

    // Check if shadow is effectively blocked
    if (luminance(payload.attenuation) < 0.01f) {
        optixTerminateRay();  // Shadow ray blocked
    } else {
        optixIgnoreIntersection();  // Continue through glass
    }
}
```

### Caustics Rendering

Caustics are **concentrated light patterns** created by curved glass:

#### Method 1: Photon Mapping
```cuda
// Photon emission pass
void emit_photons() {
    for (int photon = 0; photon < num_photons; ++photon) {
        Ray ray = generate_light_ray();
        trace_photon(ray);
        // Store photon hits in spatial data structure
    }
}

// Rendering pass - gather nearby photons
float3 gather_caustics(float3 hit_point) {
    float3 caustic_light = {0, 0, 0};
    float radius = 0.1f;

    // Find photons within radius
    for (Photon p : nearby_photons(hit_point, radius)) {
        float distance = length(p.position - hit_point);
        float weight = kernel_weight(distance, radius);
        caustic_light += weight * p.energy;
    }

    return caustic_light / (PI * radius * radius);
}
```

#### Method 2: Path Tracing with MIS
```cuda
// Multiple Importance Sampling for caustics
float3 trace_with_caustics() {
    // Standard path tracing
    float3 direct = trace_direct_lighting();

    // Add specular paths from light sources
    float3 caustic = 0;
    if (surface_is_diffuse && depth > 0) {
        // Check if previous bounce was specular
        if (prev_was_specular) {
            // This could be a caustic path
            caustic = evaluate_caustic_contribution();
        }
    }

    return direct + caustic;
}
```

### Nested Dielectrics

Handle **glass within glass** or **bubbles**:

```cuda
struct DielectricStack {
    float ior_stack[MAX_NESTED_DEPTH];
    int stack_size;
};

float get_ior_ratio(DielectricStack& stack, bool entering) {
    if (entering) {
        // Push new IOR
        float prev_ior = (stack.stack_size > 0) ?
                         stack.ior_stack[stack.stack_size-1] : 1.0f;
        stack.ior_stack[stack.stack_size++] = current_ior;
        return current_ior / prev_ior;
    } else {
        // Pop IOR
        stack.stack_size--;
        float next_ior = (stack.stack_size > 0) ?
                         stack.ior_stack[stack.stack_size-1] : 1.0f;
        return current_ior / next_ior;
    }
}
```

### Surface Properties

#### Frosted Glass
```cuda
// Perturb normal for rough glass
float3 rough_glass_normal(float3 smooth_normal, float roughness) {
    float3 random_dir = sample_hemisphere(random_seed);
    return normalize(mix(smooth_normal, random_dir, roughness));
}
```

#### Textured Glass
```cuda
// Apply texture to glass properties
extern "C" __global__ void __closesthit__textured_glass() {
    float2 uv = optixGetTriangleBarycentrics();

    // Sample textures
    float3 color = tex2D(glass_color_texture, uv);
    float ior = tex2D(ior_texture, uv).x;
    float roughness = tex2D(roughness_texture, uv).x;

    // Use textured values in shading
}
```

#### Thin Glass Approximation
```cuda
// For very thin glass (windows), skip volume
extern "C" __global__ void __closesthit__thin_glass() {
    // No refraction offset - ray continues straight
    // Apply Fresnel and tint only
    float fresnel = calculate_fresnel();

    // Trace straight through
    float3 transmit = traceRadianceRay(hit_point, ray_dir);

    // Add reflection
    float3 reflect = traceRadianceRay(hit_point, reflect_dir);

    result = (1 - fresnel) * tint * transmit + fresnel * reflect;
}
```

---

## 7. Performance Considerations

### Geometry Type Performance

**Hardware-Accelerated (RTX cores)**:
- Triangles: ~10-100x faster than custom primitives
- Best for complex geometry
- Uses dedicated RT cores

**Custom Primitives**:
- Sphere, box, etc: ~10x slower than triangles
- More memory efficient
- Flexible intersection logic

**Built-in Primitives**:
- Limited functionality
- Cannot handle internal rays for spheres
- Not suitable for glass

### Optimization Strategies

#### 1. Importance Sampling
```cuda
// Terminate rays with low contribution
if (payload.importance < 0.01f) {
    return cutoff_color;  // Early termination
}
```

#### 2. Russian Roulette
```cuda
// Probabilistically terminate rays
float continuation_prob = min(1.0f, luminance(throughput));
if (random() > continuation_prob) {
    return float3(0);  // Terminate
}
throughput /= continuation_prob;  // Compensate
```

#### 3. Level of Detail
```cuda
// Use simpler shading for distant objects
if (distance_to_camera > LOD_distance) {
    // Skip refraction, use simple transparency
    return lerp(background, glass_color, 0.5f);
}
```

#### 4. Adaptive Sampling
```cuda
// Reduce samples for smooth areas
float variance = estimate_pixel_variance();
int samples = (variance > threshold) ? 64 : 16;
```

### Memory Optimization

```cuda
// Pack material data efficiently
struct PackedGlass {
    // Pack float3 + float into float4
    float4 refraction_color_and_ior;  // xyz = color, w = ior
    float4 extinction_and_fresnel;    // xyz = extinction, w = fresnel_exp

    // Use half precision where possible
    half3 reflection_color;
    half shadow_attenuation;
};
```

---

## 8. Implementation Patterns

### Pattern 1: Simple Glass Sphere

```cuda
// Minimal implementation for clear glass sphere
extern "C" __global__ void __closesthit__simple_glass() {
    const float IOR = 1.5f;
    const float R0 = 0.04f;  // Fresnel at normal incidence

    float3 normal = optixGetWorldNormal();
    float3 ray_dir = optixGetWorldRayDirection();

    // Ensure normal faces ray
    if (dot(normal, ray_dir) > 0) normal = -normal;

    // Fresnel
    float cos_theta = abs(dot(normal, ray_dir));
    float fresnel = R0 + (1 - R0) * pow(1 - cos_theta, 5);

    // Reflection
    float3 reflect_dir = reflect(ray_dir, normal);
    float3 reflect_color = traceRadianceRay(hit_point + 0.001f * normal,
                                            reflect_dir);

    // Refraction
    float3 refract_dir;
    float3 refract_color = float3(0);
    if (refract(refract_dir, ray_dir, normal, IOR)) {
        refract_color = traceRadianceRay(hit_point - 0.001f * normal,
                                         refract_dir);
    }

    return fresnel * reflect_color + (1 - fresnel) * refract_color;
}
```

### Pattern 2: Production Glass with All Features

```cuda
// Full-featured glass implementation
class GlassRenderer {
    // Track ray state
    struct RayState {
        float3 origin;
        float3 direction;
        float importance;
        float distance_in_glass;
        int depth;
        DielectricStack ior_stack;
    };

    float3 shade(RayState& state, HitInfo& hit) {
        // 1. Entry/exit detection
        bool entering = detect_entry(state, hit);

        // 2. Update IOR stack
        float ior_ratio = update_ior_stack(state, entering);

        // 3. Compute Fresnel
        float fresnel = compute_fresnel(state, hit, ior_ratio);

        // 4. Trace reflection
        float3 reflection = trace_reflection(state, hit, fresnel);

        // 5. Trace refraction
        float3 refraction = trace_refraction(state, hit, ior_ratio, fresnel);

        // 6. Apply absorption
        if (!entering) {
            refraction *= compute_absorption(state.distance_in_glass);
        }

        // 7. Combine results
        return reflection + refraction;
    }
};
```

### Pattern 3: Hybrid Approach

```cuda
// Use different methods based on context
extern "C" __global__ void __closesthit__adaptive_glass() {
    float distance = length(hit_point - camera_position);

    if (distance < NEAR_DISTANCE) {
        // Full glass simulation for close objects
        return full_glass_shade();
    } else if (distance < MID_DISTANCE) {
        // Simplified glass for medium distance
        return simple_glass_shade();
    } else {
        // Alpha transparency for far objects
        return alpha_blend_shade();
    }
}
```

---

## 9. Common Problems and Solutions

### Problem 1: Black Pixels at Glass Edges

**Cause**: Total internal reflection not handled

**Solution**:
```cuda
if (!refract(refract_dir, ray_dir, normal, ior)) {
    // Total internal reflection - use reflection only
    return trace_reflection(hit_point, reflect_dir);
}
```

### Problem 2: Infinite Recursion

**Cause**: No depth limiting

**Solution**:
```cuda
if (payload.depth >= MAX_DEPTH) {
    return float3(0);  // Terminate
}
```

### Problem 3: Self-Intersection Artifacts

**Cause**: Secondary rays hit same surface

**Solution**:
```cuda
// Offset ray origins
float3 offset = scene_epsilon * normal;
float3 reflect_origin = hit_point + offset;
float3 refract_origin = hit_point - offset;
```

### Problem 4: Incorrect Normals

**Cause**: Normal pointing wrong direction

**Solution**:
```cuda
// Ensure normal points toward ray
if (dot(normal, ray_dir) > 0) {
    normal = -normal;
}
```

### Problem 5: Missing Glass Exits

**Cause**: Using built-in sphere primitives

**Solution**: Use custom intersection program (see Section 3)

### Problem 6: Dark Glass

**Cause**: Applying absorption on entry

**Solution**:
```cuda
// Only apply absorption when exiting
if (!entering && entry_distance >= 0) {
    float distance = ray_t - entry_distance;
    color *= exp(-extinction * distance);
}
```

### Problem 7: Fireflies

**Cause**: High-variance caustic paths

**Solution**:
```cuda
// Clamp extreme values
float3 clamp_fireflies(float3 color, float max_value = 10.0f) {
    float luminance = dot(color, float3(0.3, 0.6, 0.1));
    if (luminance > max_value) {
        color *= max_value / luminance;
    }
    return color;
}
```

---

## 10. Future Enhancements

### Advanced Dispersion

Implement spectral rendering for accurate dispersion:
```cuda
// Full spectral sampling
const int NUM_WAVELENGTHS = 32;
float wavelengths[NUM_WAVELENGTHS];
float3 spectrum_to_rgb(float spectrum[NUM_WAVELENGTHS]);

float3 render_spectral() {
    float spectrum[NUM_WAVELENGTHS] = {0};

    for (int i = 0; i < NUM_WAVELENGTHS; ++i) {
        float wavelength = 380 + i * (780 - 380) / NUM_WAVELENGTHS;
        float ior = cauchy_ior(wavelength);
        spectrum[i] = trace_wavelength(wavelength, ior);
    }

    return spectrum_to_rgb(spectrum);
}
```

### Polarization

Track light polarization for accurate Fresnel:
```cuda
struct PolarizedLight {
    float3 s_polarized;  // Perpendicular to plane
    float3 p_polarized;  // Parallel to plane
};

PolarizedLight fresnel_polarized(float cos_theta_i, float ior) {
    // Fresnel equations for s and p polarization
    float cos_theta_t = sqrt(1 - sin2_theta_t);

    float r_s = (n1 * cos_theta_i - n2 * cos_theta_t) /
                (n1 * cos_theta_i + n2 * cos_theta_t);

    float r_p = (n2 * cos_theta_i - n1 * cos_theta_t) /
                (n2 * cos_theta_i + n1 * cos_theta_t);

    return {r_s * r_s, r_p * r_p};
}
```

### Birefringence

Double refraction in crystals:
```cuda
struct BirefringentMaterial {
    float ordinary_ior;
    float extraordinary_ior;
    float3 optic_axis;
};

void trace_birefringent(Ray ray, BirefringentMaterial& material) {
    // Split ray into ordinary and extraordinary rays
    Ray ordinary_ray = refract_ordinary(ray, material.ordinary_ior);
    Ray extraordinary_ray = refract_extraordinary(ray,
                                                 material.extraordinary_ior,
                                                 material.optic_axis);

    // Trace both rays
    float3 color = 0.5f * trace(ordinary_ray) +
                   0.5f * trace(extraordinary_ray);
}
```

### Fluorescence

Wavelength shifting in materials:
```cuda
float3 fluorescence(float3 absorbed_light, float efficiency) {
    // Convert absorbed UV to visible light
    float uv_energy = absorbed_light.x;  // Assume x = UV
    float3 emitted = float3(0, uv_energy * efficiency, 0);  // Green
    return emitted;
}
```

### Adaptive Caustics

Focus samples on caustic regions:
```cuda
struct CausticMap {
    float importance[RESOLUTION_X][RESOLUTION_Y];

    void update(float3 hit_point, float contribution) {
        int x = world_to_map_x(hit_point);
        int y = world_to_map_y(hit_point);
        importance[x][y] += contribution;
    }

    int get_sample_count(float3 point) {
        float imp = lookup_importance(point);
        return MIN_SAMPLES + (MAX_SAMPLES - MIN_SAMPLES) * imp;
    }
};
```

---

## References

### OptiX SDK Examples
- `/SDK/optixWhitted/` - Complete glass sphere implementation
- `/SDK/cuda/geometry.cu` - Custom intersection programs
- `/SDK/cuda/shading.cu` - Glass material shaders
- `/SDK/cuda/helpers.h` - Utility functions

### NVIDIA Documentation
- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix9/guide/index.html)
- [OptiX API Reference](https://raytracing-docs.nvidia.com/optix9/api/index.html)
- [OptiX Quick Start](https://docs.nvidia.com/gameworks/content/gameworkslibrary/optix/optix_quickstart.htm)

### Academic Papers
- "An Improved Illumination Model for Shaded Display" - Turner Whitted (1980)
- "Measuring and Modeling the Appearance of Finished Wood" - Marschner et al. (2005)
- "A Microfacet-based BRDF Generator" - Ashikhmin & Shirley (2000)

### Forum Discussions
- [OptiX Developer Forums](https://forums.developer.nvidia.com/c/gaming-and-visualization-technologies/visualization/optix/167)
- "Sphere intersection with ray-distance dependent radius"
- "Custom Intersection Program - OptiX 7"
- "Glass rendering best practices"

---

## Conclusion

Proper glass rendering in OptiX requires:

1. **Custom intersection programs** - Built-in sphere primitives don't support internal rays
2. **Bidirectional ray handling** - Must detect both entry and exit
3. **Proper physics** - Fresnel, Snell's law, Beer-Lambert absorption
4. **Careful ray management** - Offset origins, track state, limit depth
5. **Performance awareness** - Balance quality vs speed

The OptiX SDK provides excellent examples (especially optixWhitted) that demonstrate production-ready glass rendering. The key insight is that glass is not just transparency - it's a complex interaction of reflection, refraction, and absorption that requires careful handling of ray state throughout the traversal.

For spherical glass objects specifically, custom intersection programs are mandatory due to the limitations of built-in primitives. This additional complexity is offset by the complete control it provides over the intersection logic, enabling accurate glass rendering with all physical effects.