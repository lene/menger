# Caustics Implementation Analysis

> **Purpose:** Technical analysis of existing PPM caustics implementation for future development.
> **Date:** 2025-11-24 (Updated with clarifications for autonomous development)
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
| Radius update | `sphere_combined.cu` | 1533-1564 | ❌ Not executing (kernel never called) |
| Radiance computation | `sphere_combined.cu` | 1583-1628 | ⚠️ Working but limited by radius bug |
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

### 2. Energy Normalization and Scale Factor ⚠️ NEEDS CALIBRATION

**Location:** `sphere_combined.cu:1612`

```cpp
const float caustic_scale = 10000.0f;  // TEMPORARY - for visibility
```

**Status (2025-11-24):**
- **Target value:** 1.0 (physics-based, no arbitrary scaling)
- **Current value:** 10000 (temporary exaggeration to make caustics visible)
- **No double-normalization bug:** Investigation confirmed energy flow is correct

**Energy Flow Analysis:**
The PPM implementation correctly applies two DIFFERENT divisions:

1. **Photon Emission** (`sphere_combined.cu:1543-1548`):
   ```cpp
   const float total_photons = static_cast<float>(dim.x * dim.y);
   photon_flux = make_float3(
       light.color[0] * light.intensity / total_photons,
       light.color[1] * light.intensity / total_photons,
       light.color[2] * light.intensity / total_photons
   );
   ```
   **Purpose:** Distributes light energy among all emitted photons.

2. **Radiance Computation** (`sphere_combined.cu:1660-1664`):
   ```cpp
   const float3 radiance = make_float3(
       hp.flux[0] / area,
       hp.flux[1] / area,
       hp.flux[2] / area
   );
   ```
   where `area = M_PIf * hp.radius * hp.radius`

   **Purpose:** Converts accumulated flux to radiance density using PPM formula `L = Φ / (π × R²)`.

**Why scale factor is needed:**
- Light intensity (500) / photon count (100k) produces very small numbers (~0.005 per photon)
- Without amplification, caustic contribution is below perceptible threshold
- Proper calibration requires comparing against PBRT reference render

**Next steps:**
- Implement spatial grid acceleration (speeds up testing)
- Generate PBRT reference image
- Empirically determine correct scale factor via SSIM comparison
- Target: physics-based rendering with scale=1.0

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

### 5. Spatial Grid Implementation Strategy

**Current Status:** Grid structures defined but not built/used (see Issue #1).

**Implementation Approach (2025-11-24):**

**Phase 1: Fixed Grid (Intermediate Step)**
- Use fixed world bounds: `grid_min = (-3, -3, -3)`, `grid_max = (3, 3, 3)`
- Grid resolution: 128³ (from `CAUSTICS_GRID_RESOLUTION`)
- Cell size: `(3 - (-3)) / 128 = 0.046875` units
- **Rationale:** Canonical scene sphere has radius 1.0, floor at Y=-2.0, caustic radius ~0.3 → all geometry fits within ±3 units

**Grid Building Steps (host-side orchestration in `OptiXWrapper.cpp`):**
1. Launch `__caustics_count_grid_cells()` kernel (already exists)
   - Assigns each hit point to grid cell
   - Atomically increments `grid_counts[cell_idx]`
2. CPU-side prefix sum: `grid_offsets[i] = sum(grid_counts[0..i-1])`
   - Alternatively: use Thrust `exclusive_scan` on device
3. Launch grid population kernel
   - Reorders hit points by grid cell using offsets
   - Or builds index array for indirect lookup
4. Modify `depositPhoton()` in photon tracing shader
   - Compute photon's grid cell
   - Lookup `grid_offsets[cell]` and `grid_counts[cell]`
   - Check only hit points in that cell (+ 26 neighbors for safety)
   - **Performance:** O(n×m) → O(n × avg_points_per_cell)

**Phase 2: Dynamic Grid (After Fixed Grid Validated)**
- GPU parallel reduction to compute bounding box from hit points
- Dynamic `grid_min`, `grid_max`, `cell_size` based on scene
- Adaptive grid resolution based on hit point density

**References:**
- Grid utility functions: `sphere_combined.cu:1020-1048`
- Original plan pseudocode: `CAUSTICS_PLAN.md:221-246`
- Host orchestration location: `OptiXWrapper.cpp:835-913` (`renderWithCaustics`)

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

## Reference Scene and Image

**Primary Reference Scene:** `optix-jni/test-resources/caustics-references/renders/canonical-caustics.pbrt`

**Status (2025-11-24):** ✅ Complete PBRT v4 scene file with all parameters specified

**Scene Parameters (authoritative source):**

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **Film** | Format | RGB EXR | `canonical-caustics.exr` |
| | Resolution | 800 × 600 | Matches our default render size |
| **Camera** | Position | (0, 1, 4) | Above and behind sphere |
| | Look at | (0, 0, 0) | Sphere center (origin) |
| | Up vector | (0, 1, 0) | Y-up coordinate system |
| | FOV | 45° | Standard perspective |
| **Integrator** | Type | SPPM | Stochastic Progressive Photon Mapping |
| | Max depth | 32 | Sufficient for glass traversal |
| | Photons/iteration | 100,000 | Matches our `DEFAULT_PHOTONS_PER_ITER` |
| **Sampler** | Type | pmj02bn | Progressive multi-jittered |
| | Pixel samples | 256 | High quality |
| **Light** | Type | **Point light** | Simple, predictable |
| | Position | (0, 10, 0) | Directly above sphere |
| | Intensity | RGB(500, 500, 500) | Bright enough for clear caustic |
| **Sphere** | Center | (0, 0, 0) | Origin |
| | Radius | 1.0 | Unit sphere |
| | Material | Dielectric | Pure glass (no absorption) |
| | IOR | 1.5 | Standard glass |
| **Floor** | Position | Y = -2.0 | Horizontal plane |
| | Size | 20 × 20 units | Large enough for caustic |
| | Material | Diffuse | Lambertian BRDF |
| | Reflectance | RGB(0.8, 0.8, 0.8) | High albedo gray |

**Expected Result:**
- Circular caustic centered at (0, -2, 0) - directly below sphere
- Caustic radius ~0.3 units (from thick lens formula)
- Peak brightness > 1.5× ambient floor illumination

**To render PBRT reference image:**
```bash
cd optix-jni/test-resources/caustics-references/renders
pbrt caustics-bdpt.pbrt                    # Renders to caustics-bdpt.png
cp caustics-bdpt.png ../pbrt-reference.png
```

**To run OptiX test (generates /tmp/caustics-test-output.png):**
```bash
# ⚠️ WARNING: NEVER CHANGE THESE PARAMETERS - they must match caustics-bdpt.pbrt exactly
sbt "project optixJni" "testOnly menger.optix.caustics.CausticsReferenceSpec -- -n Slow -z \"produce visible caustics\""
```

**Scene Parameters - DO NOT MODIFY (from caustics-bdpt.pbrt):**
- Camera: position=(0,4,8), lookAt=(0,0,0), up=(0,1,0), FOV=45°
- Light: Point at (0,10,0) with intensity 500
- Sphere: center=(0,0,0), radius=1.0, IOR=1.5, glass material
- Floor: Y=-2.0, solid gray reflectance 0.8 (NOT checkered)
- Resolution: 800×600

**Alternative (Mitsuba 3):** `render_reference.py` script available but PBRT is primary reference

## Autonomous Development Readiness

**Status (2025-11-24):** ✅ All clarifications complete - ready for autonomous implementation

**Documentation Review:**
- ✅ CAUSTICS_PLAN.md - Algorithm theory and pseudocode reviewed
- ✅ CAUSTICS_TEST_LADDER.md - Validation framework (C1-C8) reviewed
- ✅ CAUSTICS_REFERENCES.md - Reference materials cataloged
- ✅ canonical-caustics.pbrt - Complete scene parameters confirmed
- ✅ Energy flow analysis - Confirmed correct, no double-normalization bug

**Key Clarifications (2025-11-24):**
1. **Reference scene:** Complete PBRT file at `test-resources/caustics-references/renders/canonical-caustics.pbrt`
2. **Energy normalization:** Code is correct, target scale factor is 1.0, current 10000 is temporary
3. **Grid strategy:** Fixed bounds first ((-3,-3,-3) to (3,3,3)), then dynamic after validation
4. **Development approach:** Iterative phases with validation at each step

**Development Plan:**

### Phase 1: Fixed Spatial Grid (Performance Fix)
**Goal:** Accelerate photon deposition from O(n×m) to O(n × avg_points_per_cell)

**Tasks:**
1. Initialize fixed grid bounds in `OptiXWrapper.cpp:renderWithCaustics()`
   - Set `grid_min = (-3, -3, -3)`, `grid_max = (3, 3, 3)`
   - Use existing `CAUSTICS_GRID_RESOLUTION = 128`
2. Add grid building pass after hit point collection:
   - Launch `__caustics_count_grid_cells()` (already exists)
   - Compute prefix sum on `grid_counts` → `grid_offsets`
   - Optional: Launch grid population kernel for reordering
3. Modify `depositPhoton()` in `sphere_combined.cu:1239-1270`:
   - Compute photon's grid cell using `getCausticsGridCell()`
   - Query grid for hit points in cell + 26 neighbors
   - Check only those hit points instead of all
4. Validate: Render canonical scene, compare output to brute-force version (should be identical)

**Acceptance Criteria:**
- Visual output identical to brute-force version (pixel-perfect match)
- Render time reduction > 10× for typical scenes
- No caustic artifacts or missing energy

### Phase 2: Energy Calibration
**Goal:** Determine physics-based scale factor via reference comparison

**Tasks:**
1. Generate PBRT reference: `pbrt canonical-caustics.pbrt`
2. Render test image with scale=1.0: `./menger --object sphere --renderer optix --caustics`
3. If too dim, bisect search for scale factor targeting SSIM > 0.90
4. Document final scale factor and rationale

**Acceptance Criteria:**
- SSIM ≥ 0.90 vs PBRT reference
- Caustic brightness perceptually matches reference
- Energy conservation metrics (C5) within 10% of expected

### Phase 3: Test Ladder Implementation
**Goal:** Validate C1-C8 quality requirements

**Tasks:**
1. Implement analytic tests (C1-C5) in `CausticsValidationSpec.scala`
2. Implement convergence tests (C6-C7)
3. Implement reference match test (C8) using SSIM in `ReferenceMatchSpec.scala`
4. All tests passing

**Acceptance Criteria:**
- All 8 test ladder steps passing
- Documented test results in test output

### Phase 4: Dynamic Grid (Future Enhancement)
**Goal:** Adapt grid bounds to arbitrary scenes

**Tasks:**
1. GPU parallel reduction to compute hit point bounding box
2. Dynamic grid parameter computation
3. Validate on varied test scenes

**Acceptance Criteria:**
- Works correctly for scenes outside ±3 unit bounds
- Performance comparable to fixed grid for canonical scene

**Current Priority:** Phase 0 (Caustics Brightness Investigation) → then Phase 1 (Spatial Grid)

## Phase 0: Caustics Brightness Investigation (2025-11-25)

### Current Status
- ✅ **Caustics working** - PPM produces visible caustics on floor
- ⚠️ **Caustics too weak** - Average brightness ~35% of PBRT reference (0.325 vs 0.918)
- ✅ **Reference established** - `pbrt-reference.png` (1024 samples BDPT, camera at (0,4,8))
- ✅ **Test command documented** - Can render matching test images with all parameters
- ✅ **Test infrastructure** - CausticsReferenceSpec with brightness comparison tests
- ✅ **Code execution verified** - CUDA changes compile, load, and execute correctly

### Root Cause Analysis (UPDATED 2025-11-25)

**🔴 CRITICAL BUG IDENTIFIED: Radius Update Kernel Never Executes**

**The Problem:**
The CUDA kernel `__caustics_update_radii()` (sphere_combined.cu:1585-1629) is **NEVER BEING CALLED** by the host code.

**Evidence:**
1. Brightness invariant across ALL parameter changes: 10× photon count, 9× gathering radius, parameter tweaks - **ZERO effect**
2. Radius stays at 0.100 (initial value) across all iterations
3. Debug instrumentation showed kernel never executes

**Why This Breaks Everything:**
Without the radius update kernel executing:
- Flux doesn't get scaled each iteration (missing critical `flux *= ratio` step)
- Radii don't shrink progressively
- Final radiance becomes iteration-independent
- Result: Brightness constant regardless of photon count, radius, or other parameters

**Next Step to Fix:**
Add missing kernel launch in host code (OptiXWrapper.cpp) after each photon tracing iteration:
```cpp
optixLaunch(caustics_update_radii_pipeline, ...)
```

**Additional Context (Sample Count Disparity):**
- PBRT reference: 491M path samples (800×600 × 1024 samples/pixel)
- Our PPM: 1M photons (100K × 10 iterations)
- This 491:1 ratio contributes to brightness difference, but cannot be tested until radius update kernel is fixed

### Investigation Notes

**Failed Approaches Tested (2025-11-25):**
All of the following had **ZERO effect** on brightness due to the radius update kernel bug:
- Increasing photon count (1M → 10M): No change
- Increasing gathering radius (0.1 → 0.3): No change
- Adjusting MAX_PHOTON_BOUNCES, MAX_TRACE_DEPTH, scale_factor: No change

These approaches may become relevant after fixing the kernel bug.

### Success Criteria
- Fix radius update kernel invocation (priority #1)
- Caustic brightness within 20% of PBRT reference
- All existing 818 tests continue passing

## References

- [CAUSTICS_PLAN.md](./CAUSTICS_PLAN.md) - Original implementation plan
- [CAUSTICS_TEST_LADDER.md](./CAUSTICS_TEST_LADDER.md) - Validation framework
- [CAUSTICS_REFERENCES.md](./CAUSTICS_REFERENCES.md) - Reference scenes
- [docs/CAUSTICS.md](../docs/CAUSTICS.md) - User-facing documentation
