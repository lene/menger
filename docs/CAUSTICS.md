# Caustics Implementation (Progressive Photon Mapping)

## Overview

This document describes the Progressive Photon Mapping (PPM) implementation for rendering caustics - the bright patterns formed when light refracts through transparent objects like glass spheres.

## Algorithm

PPM is a multi-pass rendering algorithm:

### Phase 1: Hit Point Collection (`__raygen__hitpoints`)
- Trace camera rays to find visible diffuse surfaces (floor plane)
- Store hit points with position, normal, pixel coordinates
- Each hit point has a gather radius for collecting photons

### Phase 2: Photon Tracing (`__raygen__photons`)
- Emit photons from light sources toward the scene
- Trace photons through refractive objects (sphere)
- When photons hit diffuse surfaces, deposit energy on nearby hit points

### Phase 3: Radiance Computation (`__raygen__caustics_radiance`)
- For each hit point with accumulated flux, compute radiance estimate
- Formula: L = Φ / (π × R² × N_total)
- Add caustic contribution to rendered image

## Key Files

- `optix-jni/src/main/native/shaders/sphere_combined.cu` - CUDA/OptiX shaders
- `optix-jni/src/main/native/OptiXWrapper.cpp` - Host-side rendering loop
- `optix-jni/src/main/native/include/OptiXData.h` - Data structures (HitPoint, CausticsParams)

## Data Structures

### HitPoint
```cpp
struct HitPoint {
    float position[3];    // World position on diffuse surface
    float normal[3];      // Surface normal
    float flux[3];        // Accumulated photon flux (RGB)
    float radius;         // Current search radius
    unsigned int n;       // Number of photons accumulated
    unsigned int pixel_x, pixel_y;  // Pixel coordinates for final write
    float weight[3];      // BRDF weight
    unsigned int new_photons;  // Photons this iteration
};
```

### CausticsParams
```cpp
struct CausticsParams {
    bool enabled;
    int photons_per_iteration;
    int iterations;
    float initial_radius;
    float alpha;  // Radius reduction factor
    int current_iteration;
    HitPoint* hit_points;
    unsigned int* num_hit_points;
    // Grid structures (for spatial acceleration - TODO)
};
```

## CLI Options

```
--caustics                 Enable caustics rendering
--caustics-photons N       Photons per iteration (default: 100000, max: 10M)
--caustics-iterations N    PPM iterations (default: 10, max: 1000)
--caustics-radius F        Initial gather radius (default: 0.1, max: 10.0)
--caustics-alpha F         Radius reduction factor (default: 0.7, range: 0-1)
```

## Known Issues and TODOs

### Current Limitations

1. **Brute-force photon deposition** - O(n×m) complexity where n=photons, m=hit points
   - Very slow for high photon/hit point counts
   - Need to implement spatial hash grid for O(n) performance
   - Grid structures already defined in CausticsParams, awaiting implementation

2. **Caustic intensity** - Using scale factor of 10000×
   - Provides visible caustics without oversaturation
   - May need tuning for different scene configurations

### Future Improvements

1. **Spatial Hash Grid**
   - Build grid after hit point collection
   - Use grid for O(1) neighbor lookup during photon deposition
   - Grid structures already defined in CausticsParams

2. **Progressive Radius Reduction**
   - Implement `__caustics_update_radii` kernel
   - Shrink radius after each iteration for convergence

3. **Multiple Light Support**
   - Weight photon emission by light intensity
   - Support point lights and area lights

4. **Spectral Dispersion**
   - Trace wavelength-dependent refraction
   - Rainbow caustic effects

## Debugging Tips

1. **Verify hit point collection**: Check `[Caustics] Hit points collected: N` log message

2. **Debug photon deposition**: Temporarily modify `__raygen__caustics_radiance` to color pixels green where flux > 0

3. **Check coordinate systems**: Ensure Y-flip in `__raygen__hitpoints` matches `__raygen__rg`

## References

- Henrik Wann Jensen, "Realistic Image Synthesis Using Photon Mapping"
- Hachisuka et al., "Progressive Photon Mapping" (SIGGRAPH Asia 2008)
- OptiX Programming Guide - Custom ray generation programs
