# Caustics Implementation Analysis

> **Purpose:** Technical analysis of existing PPM caustics implementation for future development.
> **Date:** 2025-11-22
> **Branch:** Based on `feature/caustics`

## Implementation Status

### Completed Components

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| Data structures | `OptiXData.h` | 98-146 | ✅ Complete |
| Hit point generation | `sphere_combined.cu` | 1088-1205 | ✅ Working |
| Grid cell utilities | `sphere_combined.cu` | 1020-1048 | ✅ Complete |
| Photon tracing | `sphere_combined.cu` | 1277-1397 | ✅ Working |
| Photon emission | `sphere_combined.cu` | 1446-1514 | ✅ Working |
| Photon deposition | `sphere_combined.cu` | 1239-1270 | ⚠️ O(n×m) |
| Radius update | `sphere_combined.cu` | 1533-1564 | ✅ Complete |
| Radiance computation | `sphere_combined.cu` | 1583-1628 | ✅ Complete |
| JNI bindings | `JNIBindings.cpp` | - | ✅ Complete |
| Scala integration | `OptiXRenderer.scala` | 158-170 | ✅ Complete |
| CLI options | `MengerCLIOptions.scala` | - | ✅ Complete |

### Data Structures

```cpp
// From OptiXData.h

struct HitPoint {
    float position[3];      // World position on diffuse surface
    float normal[3];        // Surface normal at hit point
    float flux[3];          // Accumulated photon flux (RGB)
    float radius;           // Current search radius (shrinks over iterations)
    unsigned int n;         // Number of photons accumulated so far
    unsigned int pixel_x;   // Pixel X coordinate for final write
    unsigned int pixel_y;   // Pixel Y coordinate for final write
    float weight[3];        // BRDF weight for this view direction
    unsigned int new_photons; // Photons accumulated this iteration
    float pad;              // Alignment padding
};

struct CausticsParams {
    bool enabled;
    int photons_per_iteration;
    int iterations;
    float initial_radius;
    float alpha;                    // Radius reduction factor (0.7 typical)
    int current_iteration;

    // GPU buffers
    HitPoint* hit_points;
    unsigned int* num_hit_points;   // Atomic counter

    // Spatial grid (DEFINED but NOT IMPLEMENTED)
    unsigned int* grid_counts;
    unsigned int* grid_offsets;
    float grid_min[3];
    float grid_max[3];
    float cell_size;
    unsigned int grid_resolution;   // 128

    unsigned long long total_photons_traced;
};
```

### Constants

```cpp
// From OptiXConstants.h
constexpr int MAX_HIT_POINTS = 2000000;
constexpr int DEFAULT_PHOTONS_PER_ITER = 100000;
constexpr int DEFAULT_CAUSTICS_ITERATIONS = 10;
constexpr float DEFAULT_INITIAL_RADIUS = 0.1f;
constexpr float DEFAULT_PPM_ALPHA = 0.7f;
constexpr int CAUSTICS_GRID_RESOLUTION = 128;
constexpr int MAX_PHOTON_BOUNCES = 10;
```

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Hit Point Collection (__raygen__hitpoints)             │
│ - Trace camera rays to find visible floor pixels                │
│ - Store HitPoint with position, normal, pixel coords            │
│ - Initialize radius = initial_radius                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: PPM Iterations (loop N times)                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 2a: Photon Emission (__raygen__photons)                     │ │
│ │ - Emit photons from light toward sphere                     │ │
│ │ - Trace through sphere (refraction + Beer-Lambert)          │ │
│ │ - Deposit on hit points via depositPhoton()                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 2b: Radius Update (__caustics_update_radii)                 │ │
│ │ - R_new = R * sqrt((N + α*M) / (N + M))                     │ │
│ │ - Scale flux by same ratio                                  │ │
│ │ - Reset new_photons counter                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Radiance Computation (__raygen__caustics_radiance)     │
│ - L = flux / (π × R² × N_total)                                 │
│ - Scale by 10000× (magic number for visibility)                 │
│ - Add to existing pixel color (additive blending)               │
└─────────────────────────────────────────────────────────────────┘
```

## Known Issues

### 1. Brute-Force Photon Deposition (CRITICAL)

**Location:** `sphere_combined.cu:1239-1270`

```cpp
// Current implementation - O(n × m) where n=photons, m=hit_points
for (unsigned int hp_idx = 0; hp_idx < num_hp; hp_idx += 1) {
    HitPoint& hp = params.caustics.hit_points[hp_idx];
    // Check distance, deposit if within radius
}
```

**Impact:** For 100k photons × 500k hit points = 50 billion distance checks per iteration.

**Solution:** Implement spatial hash grid (structures already defined):
- `getCausticsGridCell()` - Already implemented
- `getCausticsGridIndex()` - Already implemented
- `__caustics_count_grid_cells()` - Already implemented
- Missing: Grid building pass and grid-accelerated lookup in `depositPhoton()`

### 2. ~~Magic Intensity Scale Factor~~ ✅ FIXED

**Location:** `sphere_combined.cu:1672`

```cpp
const float caustic_scale = 1.0f;  // Physics-based
```

**Fix (2025-11-22):** Removed double-normalization bug. The flux was being divided by `total_photons` twice (during emission AND radiance computation). Now uses correct PPM formula: `L = Φ / (π × R²)`.

### 3. ~~No Energy Tracking~~ ✅ IMPLEMENTED

**Fix (2025-11-22):** Added atomic counters throughout the pipeline tracking:
- `photons_emitted`, `total_flux_emitted` (C1)
- `sphere_hits`, `sphere_misses` (C2)
- `refraction_events`, `tir_events` (C3)
- `photons_deposited`, `total_flux_deposited` (C4)
- `total_flux_absorbed` (C5)
- `max_caustic_brightness` (C7)

Stats accessible via `OptiXRenderer.getCausticsStats()`.

### 4. ~~No Convergence Metrics~~ ✅ IMPLEMENTED

**Fix (2025-11-22):** `CausticsStats` struct now tracks:
- `avg_radius`, `min_radius`, `max_radius`
- `flux_variance`
- Timing metrics for each phase

### 5. Brute-Force Performance (REMAINING CRITICAL ISSUE)

The spatial hash grid is the main remaining work. See Issue #1 above.

## Test Ladder Status (Updated 2025-11-22)

| Step | Quality Goal | Status | Notes |
|------|--------------|--------|-------|
| C1 | Emission count/distribution | ✅ Instrumented | `photons_emitted`, `total_flux_emitted` tracked |
| C2 | Sphere hit rate | ✅ Instrumented | `sphere_hits`, `sphere_misses` tracked |
| C3 | Refraction accuracy | ✅ Instrumented | `refraction_events`, `tir_events` tracked |
| C4 | Focal point position | ✅ Instrumented | `photons_deposited`, `hit_points_with_flux` tracked |
| C5 | Energy conservation | ✅ Instrumented | `total_flux_emitted/deposited/absorbed` tracked |
| C6 | PPM convergence | ✅ Instrumented | `avg_radius`, `flux_variance` tracked |
| C7 | Caustic brightness | ✅ Instrumented | `max_caustic_brightness` tracked |
| C8 | Reference match | ✅ Reference ready | Mitsuba reference at `src/test/resources/caustics-reference.png` |

**Next Steps:** Run `CausticsValidationSpec` tests to verify instrumentation works, then validate C1-C8 quality goals.

## Files Modified (2025-11-22)

### Instrumentation (Complete)
- `sphere_combined.cu` - Added atomic counters in shaders ✅
- `OptiXData.h` - CausticsStats struct (already existed) ✅
- `OptiXWrapper.cpp` - Stats buffer allocation and getCausticsStats() ✅
- `JNIBindings.cpp` - getCausticsStatsNative() JNI binding ✅

### Tests (Ready to Run)
- `CausticsValidationSpec.scala` - Analytic tests for C1-C5 ✅
- `ReferenceMatchSpec.scala` - Image comparison for C8 ✅
- `ImageComparison.scala` - SSIM implementation ✅

### Remaining Work: Spatial Grid Performance Fix
- `sphere_combined.cu` - Implement grid-accelerated `depositPhoton()` for O(1) lookup
- `OptiXWrapper.cpp` - Add grid building pass before photon tracing

## Reference Image

**Location:** `optix-jni/src/test/resources/caustics-reference.png`

**Generated with:** Mitsuba 3 (`pip install mitsuba`)

**Render script:** `optix-jni/test-resources/caustics-references/render_reference.py`

**Scene parameters (matching arc42 canonical scene):**
- Glass sphere: center=(0,0,0), radius=1.0, IOR=1.5
- Floor: Y=-2.0, diffuse gray (0.8)
- Light: Area light at (0,10,0), 2x2 size, intensity 5000
- Camera: (0,1,4) looking at origin, FOV=45°
- Resolution: 800x600, 4096 samples

**To regenerate:**
```bash
cd optix-jni/test-resources/caustics-references
pip install mitsuba numpy pillow
python3 render_reference.py
cp output/canonical-caustics-reference.png ../../src/test/resources/caustics-reference.png
```

## References

- [CAUSTICS_PLAN.md](./CAUSTICS_PLAN.md) - Original implementation plan
- [CAUSTICS_TEST_LADDER.md](./CAUSTICS_TEST_LADDER.md) - Validation framework
- [CAUSTICS_REFERENCES.md](./CAUSTICS_REFERENCES.md) - Reference scenes
- [docs/CAUSTICS.md](../docs/CAUSTICS.md) - User-facing documentation
