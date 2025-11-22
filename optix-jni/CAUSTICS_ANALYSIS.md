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

### 2. Magic Intensity Scale Factor

**Location:** `sphere_combined.cu:1612`

```cpp
const float caustic_scale = 10000.0f;
```

**Issue:** This is a tuned constant, not physics-based. The PPM radiance formula should produce correct values without scaling.

**Root cause candidates:**
- Photon flux not properly normalized by total photon count
- Area term (π × R²) may have units mismatch
- Light intensity not calibrated

### 3. No Energy Tracking

**Issue:** Cannot validate C5 (energy conservation) without tracking:
- Total flux emitted
- Total flux deposited
- Total flux absorbed (Beer-Lambert)
- Total flux reflected (Fresnel)

### 4. No Convergence Metrics

**Issue:** Cannot validate C6 (convergence) without tracking:
- Per-iteration variance
- Radius distribution
- Flux accumulation rate

## Test Ladder Gaps

| Step | Quality Goal | Existing Code | Test Gap |
|------|--------------|---------------|----------|
| C1 | Emission count/distribution | `__raygen__photons` | No count validation |
| C2 | Sphere hit rate | `tracePhoton` | No geometric validation |
| C3 | Refraction accuracy | `tracePhoton` (Snell's law) | No angle validation |
| C4 | Focal point position | `depositPhoton` | No centroid validation |
| C5 | Energy conservation | Not tracked | Need instrumentation |
| C6 | PPM convergence | Not measured | Need metrics |
| C7 | Caustic brightness | Not measured | Need comparison |
| C8 | Reference match | Not implemented | Need PBRT reference |

## Files to Modify

### For Analytic Tests (C1-C5)
- `optix-jni/src/test/scala/menger/optix/caustics/` - New test files

### For Instrumentation
- `OptiXData.h` - Add CausticsStats struct
- `sphere_combined.cu` - Add atomic counters in shaders
- `OptiXWrapper.cpp` - Expose stats via JNI
- `OptiXRenderer.scala` - Add stats retrieval methods

### For Spatial Grid Fix
- `sphere_combined.cu` - Implement grid-accelerated `depositPhoton()`
- `OptiXWrapper.cpp` - Add grid building pass

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
