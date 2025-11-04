# Fix Plan for OptiX Volume Absorption

## Root Cause (CONFIRMED Nov 5, 2025)
The OptiX built-in sphere primitive does NOT support ray intersection from inside the sphere. When a refracted ray is traced from inside the sphere, it never hits the back surface, preventing the exit code path from executing.

## Definitive Evidence
Comprehensive testing with explicit hit_t value checks confirms:
- **Test 1**: Ray from sphere center (tmin=0) → hit_t=-1.0 (NO intersection)
- **Test 2**: Ray from just inside surface (tmin=0.001) → hit_t=-1.0 (NO intersection)
- **Test 3**: Same ray with tmin=0 → hit_t=-1.0 (tmin not the issue)
- **Conclusion**: This is a fundamental limitation of OptiX sphere primitives

## Solution: Compute Exit Point Analytically

When a ray enters the sphere and refracts:
1. Compute the exit point analytically using ray-sphere intersection math
2. Calculate the travel distance through the sphere
3. Apply Beer-Lambert absorption immediately
4. Trace the continuation ray from the exit point

### Implementation Steps

1. **Add analytical exit computation**:
   - When entering, compute where refracted ray exits sphere
   - Use quadratic formula for ray-sphere intersection
   - Take the far intersection point

2. **Apply absorption immediately**:
   - Calculate travel distance = exit_t - entry_t
   - Apply Beer-Lambert law: I = I₀ * exp(-α * d)
   - Modify the color before tracing continuation ray

3. **Trace from exit point**:
   - Start the refracted ray from the computed exit point
   - This avoids the need for a second intersection

### Code Changes

In `__closesthit__ch`, when `entering == true`:
```cuda
// Compute where refracted ray exits sphere
float exit_t = compute_sphere_exit(refract_origin, refract_dir, sphere_center, radius);
float travel_distance = exit_t;

// Apply absorption
float3 absorption = exp(-alpha * travel_distance);
// Apply to the color that comes back from the refracted ray

// Trace from exit point
float3 exit_point = refract_origin + refract_dir * exit_t;
optixTrace from exit_point...
```

This avoids relying on OptiX to detect the exit and ensures absorption is always applied.