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

**To render reference image:**
```bash
cd optix-jni/test-resources/caustics-references/renders
pbrt canonical-caustics.pbrt                    # Renders to canonical-caustics.exr
imgtool convert canonical-caustics.exr canonical-caustics.png
cp canonical-caustics.png ../../../src/test/resources/caustics-reference.png
```

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
- ⚠️ **Caustics too weak** - Significantly dimmer/smaller than PBRT reference
- ✅ **Reference established** - `pbrt-reference.png` (1024 samples BDPT, camera at (0,4,8))
- ✅ **Test command documented** - Can render matching test images with all parameters

### Root Cause Analysis

**Identified Issues (Priority Order):**

1. **Max photon bounces too low** (10 vs PBRT's 64)
   - Location: `OptiXData.h:72` - `MAX_PHOTON_BOUNCES = 10`
   - Impact: Photons with complex internal reflections through glass sphere terminated early
   - Energy loss before reaching floor reduces caustic brightness

2. **Max camera ray depth too low** (5 vs PBRT's 64)
   - Location: `OptiXData.h:12` - `MAX_TRACE_DEPTH = 5`
   - Impact: Limited indirect illumination capture from caustic-lit surfaces

3. **Scale factor possibly too aggressive** (50.0)
   - Location: `sphere_combined.cu:1666` - `const float scale_factor = 50.0f`
   - Current empirical scaling may be suppressing caustic brightness
   - Need to compare against physics-based reference

4. **Photon count** (1M total = 100k × 10 iterations)
   - May need more photons for smoother, brighter caustics convergence

5. **Radiance normalization formula**
   - Current: `flux / (area × iterations × scale_factor)`
   - Division by iterations may be double-counting the PPM averaging

### Test-Driven Improvement Strategy

**Philosophy:** Ensure correctness first, optimize performance later. Add tests at each step to prevent regressions.

**Phase 1: Test Infrastructure**
- Create caustics brightness comparison test
  - Load reference image (`pbrt-reference.png`)
  - Render test image with matching parameters
  - Compare caustic region brightness/intensity
  - Assert within acceptable range (±20%)
- Baseline correctness tests
  - Caustic exists (non-zero brightness in expected region)
  - Scene geometry correct (sphere size, floor position)
  - Glass refraction working (IOR 1.5 behavior visible)

**Phase 2: Systematic Improvements** (each with commit + tests)

1. **Increase MAX_PHOTON_BOUNCES: 10 → 32**
   - Change: `OptiXData.h:72`
   - Test: Verify caustic brightness increases vs baseline
   - Commit: "Increase photon bounces from 10 to 32 for better caustics"

2. **Increase MAX_TRACE_DEPTH: 5 → 16**
   - Change: `OptiXData.h:12`
   - Test: Verify indirect illumination capture improves
   - Commit: "Increase camera ray depth from 5 to 16"

3. **Reduce/remove scale factor: 50.0 → 10.0 or remove**
   - Change: `sphere_combined.cu:1666`
   - Test: Verify caustic brightness closer to reference
   - Commit: "Adjust scale factor for correct caustic brightness"

4. **Investigate radiance normalization**
   - Add debug logging for flux values, area, iterations
   - Review formula at `sphere_combined.cu:1666-1674`
   - Test: Verify energy conservation
   - Commit: "Fix radiance normalization formula"

5. **Increase photon count** (if needed)
   - Change: CLI default or recommend higher values
   - Test: Verify smoother caustics with more samples
   - Commit: "Increase default photon count"

**Phase 3: Validation & Documentation**
1. Run full test suite - ensure all 818 tests pass
2. Visual comparison - test image vs reference
3. Update documentation - document final parameters
4. Commit reference files - `pbrt-reference.png`, `reference-scene.pbrt`

### Build Requirements
After each C++/CUDA change:
```bash
rm -rf optix-jni/target/native
sbt "project optixJni" compile
sbt "project optixJni" test
```

### Success Criteria
- Caustic brightness within 20% of PBRT reference
- All existing 818 tests continue passing
- New tests document expected caustic behavior
- Each improvement step has test coverage

## References

- [CAUSTICS_PLAN.md](./CAUSTICS_PLAN.md) - Original implementation plan
- [CAUSTICS_TEST_LADDER.md](./CAUSTICS_TEST_LADDER.md) - Validation framework
- [CAUSTICS_REFERENCES.md](./CAUSTICS_REFERENCES.md) - Reference scenes
- [docs/CAUSTICS.md](../docs/CAUSTICS.md) - User-facing documentation
