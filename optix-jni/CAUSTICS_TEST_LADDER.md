# Caustics Test Ladder

> **Purpose:** Step-by-step validation framework for Progressive Photon Mapping (PPM) caustics.
> Each step must pass before proceeding to the next. This ensures systematic debugging
> and prevents wasted effort on downstream issues caused by upstream bugs.
>
> **Reference:** [arc42 Section 10 - Quality Requirements](../docs/arc42/10-quality-requirements.md#caustics-quality-progressive-photon-mapping)

## Overview

```
Step 8: Reference Match ──────── C8: SSIM > 0.90
    ▲
Step 7: Brightness ───────────── C7: Peak > 1.5× ambient
    ▲
Step 6: Convergence ──────────── C6: Variance decreases
    ▲
Step 5: Energy Conservation ──── C5: Flux in ≈ flux out
    ▲
Step 4: Focal Point ──────────── C4: Centroid at predicted position
    ▲
Step 3: Refraction ───────────── C3: Exit angles match Snell's law
    ▲
Step 2: Sphere Hits ──────────── C2: Hit rate matches geometry
    ▲
Step 1: Emission ─────────────── C1: Correct count & distribution
```

## Canonical Test Scene

All tests use this fixed configuration for reproducibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sphere center | (0, 0, 0) | Origin for simple geometry |
| Sphere radius | 1.0 | Unit sphere |
| Sphere IOR | 1.5 | Standard glass |
| Plane axis | Y | Horizontal floor |
| Plane position | -2.0 | Below sphere, within focal range |
| Light direction | (0, -1, 0) | Vertical, creates circular caustic |
| Camera position | (0, 1, 4) | Above and behind, sees caustic |

**Expected caustic:** Circular bright spot centered at (0, -2, 0) with radius ~0.3 units.

---

## Step 1: Photon Emission (C1)

### Goal
Verify the photon emitter produces the correct number of photons with uniform directional distribution.

### Analytic Validation

**Test 1.1: Photon Count**
```
Given: Request N photons
When: Emission kernel runs
Then: Exactly N photons are generated
```

**Test 1.2: Uniform Distribution**
```
Given: N photons emitted from point light
When: Directions are binned into hemisphere sectors
Then: Each sector contains N/num_sectors ± 5% photons
```

**Test 1.3: Unit Direction Vectors**
```
Given: All emitted photon directions
When: Vector magnitude computed
Then: |direction| = 1.0 ± 1e-6 for all photons
```

### Implementation

```cpp
// Test struct to capture emission data
struct EmissionTestData {
    int photon_count;
    float3* directions;
    float3* origins;
};

// Validation function
bool validateEmission(EmissionTestData& data, int expected_count) {
    // 1. Count check
    if (data.photon_count != expected_count) return false;

    // 2. Unit vector check
    for (int i = 0; i < data.photon_count; i++) {
        float mag = length(data.directions[i]);
        if (abs(mag - 1.0f) > 1e-6f) return false;
    }

    // 3. Distribution check (bin into 6 hemisphere directions)
    int bins[6] = {0}; // +x, -x, +y, -y, +z, -z
    for (int i = 0; i < data.photon_count; i++) {
        float3 d = data.directions[i];
        int max_axis = (abs(d.x) > abs(d.y)) ?
                       ((abs(d.x) > abs(d.z)) ? 0 : 2) :
                       ((abs(d.y) > abs(d.z)) ? 1 : 2);
        int bin = max_axis * 2 + ((&d.x)[max_axis] > 0 ? 0 : 1);
        bins[bin]++;
    }

    float expected_per_bin = expected_count / 6.0f;
    for (int i = 0; i < 6; i++) {
        if (abs(bins[i] - expected_per_bin) > expected_per_bin * 0.10f) {
            // Allow 10% tolerance for small sample sizes
            return false;
        }
    }

    return true;
}
```

### Scala Test Specification

```scala
class PhotonEmissionSpec extends AnyFlatSpec with Matchers {
  "Photon emitter" should "emit exactly the requested number of photons" in {
    val requested = 10000
    val emitted = CausticsModule.emitPhotons(requested)
    emitted.count shouldBe requested
  }

  it should "produce uniformly distributed directions" in {
    val photons = CausticsModule.emitPhotons(60000)
    val bins = photons.directions.groupBy(directionToBin)
    bins.values.foreach { bin =>
      bin.size shouldBe (10000 +- 1000) // 10% tolerance
    }
  }

  it should "produce unit direction vectors" in {
    val photons = CausticsModule.emitPhotons(1000)
    photons.directions.foreach { d =>
      d.magnitude shouldBe (1.0 +- 1e-6)
    }
  }
}
```

### Pass Criteria
- [ ] Photon count matches exactly
- [ ] All direction vectors are unit length
- [ ] Distribution uniform within 10% per bin

---

## Step 2: Sphere Hit Rate (C2)

### Goal
Verify photons correctly intersect the sphere at the expected geometric rate.

### Analytic Calculation

For a unit sphere at origin with light direction (0, -1, 0):
- Cross-sectional area: π × r² = π × 1² = π
- Hemisphere solid angle: 2π steradians
- Expected hit rate for downward-pointing photons: ~50% of those in the -Y hemisphere

More precisely, for uniform emission over full sphere:
- Photons traveling toward sphere (from above): ~25% of total
- Of those, fraction hitting unit sphere from distance d: (r²)/(d² + r²)

For our canonical scene (light at y=+10, sphere at origin):
- Solid angle subtended by sphere: 2π(1 - cos(θ)) where θ = arctan(1/10)
- Expected hit rate: ~0.5% for distant light, higher for closer light

**Simplified Test:** Point light at (0, 5, 0), emit only downward (-Y hemisphere):
- Expected hit rate = (sphere_radius / distance)² × π/4 ≈ 3.1%

### Implementation

```cpp
struct HitTestData {
    int total_photons;
    int sphere_hits;
    float3 sphere_center;
    float sphere_radius;
    float3 light_position;
};

float expectedHitRate(float3 light_pos, float3 sphere_center, float radius) {
    float distance = length(light_pos - sphere_center);
    // Solid angle of sphere as seen from light
    float sin_theta = radius / distance;
    float cos_theta = sqrt(1.0f - sin_theta * sin_theta);
    // Fraction of hemisphere
    return (1.0f - cos_theta) / 2.0f;
}

bool validateHitRate(HitTestData& data) {
    float expected = expectedHitRate(
        data.light_position,
        data.sphere_center,
        data.sphere_radius
    );
    float actual = (float)data.sphere_hits / data.total_photons;

    // Allow 5% relative error
    return abs(actual - expected) < expected * 0.05f;
}
```

### Scala Test Specification

```scala
class SphereHitRateSpec extends AnyFlatSpec with Matchers {
  val canonicalScene = CausticsTestScene.canonical

  "Photon tracer" should "hit the sphere at the expected geometric rate" in {
    val photons = CausticsModule.emitPhotons(100000)
    val hits = CausticsModule.traceToSphere(photons, canonicalScene.sphere)

    val expectedRate = calculateExpectedHitRate(
      canonicalScene.lightPosition,
      canonicalScene.sphere
    )

    val actualRate = hits.size.toDouble / photons.count
    actualRate shouldBe (expectedRate +- expectedRate * 0.05)
  }

  private def calculateExpectedHitRate(lightPos: Vec3, sphere: Sphere): Double = {
    val distance = (lightPos - sphere.center).magnitude
    val sinTheta = sphere.radius / distance
    val cosTheta = math.sqrt(1.0 - sinTheta * sinTheta)
    (1.0 - cosTheta) / 2.0
  }
}
```

### Pass Criteria
- [ ] Hit rate within 5% of geometric prediction
- [ ] No false positives (hits outside sphere bounds)
- [ ] Hit positions lie on sphere surface (distance from center = radius)

---

## Step 3: Refraction Accuracy (C3)

### Goal
Verify Snell's law is correctly implemented at sphere entry and exit.

### Analytic Calculation

**Snell's Law:** n₁ sin(θ₁) = n₂ sin(θ₂)

For our canonical scene:
- IOR of glass (n₂) = 1.5
- IOR of air (n₁) = 1.0
- Entry: sin(θ₂) = sin(θ₁) / 1.5
- Exit: sin(θ₂) = 1.5 × sin(θ₁)

**Test Cases:**

| Entry Angle (θ₁) | Expected Refracted (θ₂) |
|------------------|-------------------------|
| 0° (normal incidence) | 0° |
| 30° | 19.47° |
| 45° | 28.13° |
| 60° | 35.26° |

For exit (reverse, glass to air):

| Entry Angle (θ₁) | Expected Refracted (θ₂) |
|------------------|-------------------------|
| 0° | 0° |
| 19.47° | 30° |
| 28.13° | 45° |
| 35.26° | 60° |
| > 41.8° | Total Internal Reflection |

### Implementation

```cpp
// Returns refracted direction, or zero vector for TIR
float3 refract(float3 incident, float3 normal, float eta) {
    float cos_i = -dot(incident, normal);
    float sin2_t = eta * eta * (1.0f - cos_i * cos_i);

    if (sin2_t > 1.0f) {
        return make_float3(0, 0, 0); // TIR
    }

    float cos_t = sqrt(1.0f - sin2_t);
    return eta * incident + (eta * cos_i - cos_t) * normal;
}

struct RefractionTestCase {
    float3 incident;
    float3 normal;
    float eta;
    float3 expected_refracted;
    float tolerance_rad;
};

bool validateRefraction(RefractionTestCase& tc) {
    float3 actual = refract(tc.incident, tc.normal, tc.eta);
    float angle_error = acos(dot(normalize(actual), normalize(tc.expected_refracted)));
    return angle_error < tc.tolerance_rad;
}
```

### Scala Test Specification

```scala
class RefractionAccuracySpec extends AnyFlatSpec with Matchers {
  val tolerance = 0.01 // radians, ~0.57 degrees

  "Snell's law implementation" should "correctly refract at normal incidence" in {
    val incident = Vec3(0, -1, 0)
    val normal = Vec3(0, 1, 0)
    val refracted = CausticsModule.refract(incident, normal, 1.0 / 1.5)

    angleError(refracted, Vec3(0, -1, 0)) should be < tolerance
  }

  it should "correctly refract at 30 degrees" in {
    val theta1 = math.toRadians(30)
    val incident = Vec3(math.sin(theta1), -math.cos(theta1), 0).normalized
    val normal = Vec3(0, 1, 0)

    val refracted = CausticsModule.refract(incident, normal, 1.0 / 1.5)

    val theta2 = math.asin(math.sin(theta1) / 1.5) // 19.47°
    val expected = Vec3(math.sin(theta2), -math.cos(theta2), 0).normalized

    angleError(refracted, expected) should be < tolerance
  }

  it should "handle total internal reflection at critical angle" in {
    val criticalAngle = math.asin(1.0 / 1.5) // ~41.8°
    val incident = Vec3(math.sin(criticalAngle + 0.1), -math.cos(criticalAngle + 0.1), 0)
    val normal = Vec3(0, 1, 0)

    val result = CausticsModule.refract(incident, normal, 1.5 / 1.0) // glass to air
    result shouldBe None // TIR
  }

  private def angleError(v1: Vec3, v2: Vec3): Double = {
    math.acos((v1.normalized dot v2.normalized).clamp(-1, 1))
  }
}
```

### Pass Criteria
- [ ] Entry refraction angles within 0.01 rad of Snell's law prediction
- [ ] Exit refraction angles within 0.01 rad of Snell's law prediction
- [ ] TIR correctly detected at angles > critical angle (41.8° for glass)

---

## Step 4: Focal Point Position (C4)

### Goal
Verify photons converge at the predicted focal point.

### Analytic Calculation

For a sphere acting as a lens (paraxial approximation):
- **Lensmaker's equation:** 1/f = (n-1)(1/R₁ - 1/R₂)
- For sphere: R₁ = R (entry), R₂ = -R (exit, negative for convex-convex)
- **Focal length:** 1/f = (n-1)(1/R - (-1/R)) = (n-1)(2/R)
- f = R / (2(n-1))

For our canonical scene:
- R = 1.0, n = 1.5
- f = 1.0 / (2 × 0.5) = 1.0

**Back focal distance** (from sphere center):
- BFD = f + R = 2.0 (for thin lens approximation)

More accurately for thick lens (sphere), the focal point is approximately:
- Distance from sphere center ≈ R × n / (2(n-1)) = 1.5 / 1.0 = 1.5 units

So the caustic should form around y = -1.5 (below sphere center at y=0).

However, this is approximate. The plane at y = -2 will show a defocused caustic ring.

### Implementation

```cpp
struct FocalPointTestData {
    int num_photons;
    float3* hit_positions;  // Where photons hit the floor plane
    float3 predicted_focal_point;
    float tolerance;
};

float3 computeCentroid(float3* positions, int count) {
    float3 sum = make_float3(0, 0, 0);
    for (int i = 0; i < count; i++) {
        sum = sum + positions[i];
    }
    return sum / (float)count;
}

bool validateFocalPoint(FocalPointTestData& data) {
    float3 centroid = computeCentroid(data.hit_positions, data.num_photons);
    float error = length(centroid - data.predicted_focal_point);
    return error < data.tolerance;
}
```

### Scala Test Specification

```scala
class FocalPointSpec extends AnyFlatSpec with Matchers {
  val canonicalScene = CausticsTestScene.canonical

  "Caustic focal point" should "be at predicted position" in {
    val photons = CausticsModule.emitPhotons(100000)
    val refracted = CausticsModule.traceThrough(photons, canonicalScene.sphere)
    val floorHits = CausticsModule.intersectPlane(refracted, canonicalScene.floor)

    val centroid = computeCentroid(floorHits.positions)
    val predicted = predictFocalPoint(canonicalScene)

    (centroid - predicted).magnitude should be < 0.1 // 0.1 unit tolerance
  }

  it should "produce a circular pattern for vertical light" in {
    val photons = CausticsModule.emitPhotons(100000)
    val floorHits = CausticsModule.fullTrace(photons, canonicalScene)

    val centroid = computeCentroid(floorHits.positions)
    val distances = floorHits.positions.map(p => (p - centroid).magnitude)
    val stdDev = standardDeviation(distances)

    // Circular pattern should have relatively uniform radial distribution
    stdDev should be < 0.5
  }

  private def predictFocalPoint(scene: CausticsTestScene): Vec3 = {
    // Thick lens formula for sphere
    val n = scene.sphere.ior
    val r = scene.sphere.radius
    val focalDistance = r * n / (2 * (n - 1))
    Vec3(scene.sphere.center.x, scene.sphere.center.y - focalDistance, scene.sphere.center.z)
  }
}
```

### Pass Criteria
- [ ] Centroid within 0.1 units of predicted focal point
- [ ] Pattern is approximately circular (low standard deviation in radial distances)
- [ ] No photons "leak" to unexpected locations

---

## Step 5: Energy Conservation (C5)

### Goal
Verify total photon energy is conserved through the system (accounting for absorption).

### Analytic Calculation

**Energy budget:**
- Input: N photons × initial_power_per_photon
- Absorbed: Beer-Lambert absorption during transit through glass
- Fresnel losses: Reflected photons at interfaces
- Output: Deposited energy on floor

For our canonical scene (clear glass, no absorption):
- Fresnel reflection at n=1.5: ~4% per interface (at normal incidence)
- Total transmission: ~92% (two interfaces)
- Energy deposited should be ~92% of incident energy (±5% for statistical variance)

**Beer-Lambert absorption:**
- For colored glass with absorption coefficient α:
- Transmission = exp(-α × distance)
- Distance through sphere ≈ 2R = 2.0 for normal incidence

### Implementation

```cpp
struct EnergyTestData {
    float total_input_energy;
    float total_deposited_energy;
    float total_absorbed_energy;
    float total_reflected_energy;
};

bool validateEnergyConservation(EnergyTestData& data) {
    float total_output = data.total_deposited_energy +
                         data.total_absorbed_energy +
                         data.total_reflected_energy;

    float error = abs(total_output - data.total_input_energy) / data.total_input_energy;
    return error < 0.05f; // 5% tolerance
}

// For clear glass (no absorption), check Fresnel losses only
bool validateFresnelOnly(EnergyTestData& data, float expected_transmission) {
    float actual_transmission = data.total_deposited_energy / data.total_input_energy;
    float error = abs(actual_transmission - expected_transmission);
    return error < 0.05f;
}
```

### Scala Test Specification

```scala
class EnergyConservationSpec extends AnyFlatSpec with Matchers {
  val canonicalScene = CausticsTestScene.canonical

  "Energy conservation" should "balance input and output within 5%" in {
    val result = CausticsModule.fullTraceWithEnergy(100000, canonicalScene)

    val totalInput = result.inputEnergy
    val totalOutput = result.depositedEnergy + result.absorbedEnergy + result.reflectedEnergy

    val error = math.abs(totalOutput - totalInput) / totalInput
    error should be < 0.05
  }

  it should "transmit ~92% through clear glass (Fresnel losses only)" in {
    val result = CausticsModule.fullTraceWithEnergy(100000, canonicalScene)

    val transmission = result.depositedEnergy / result.inputEnergy
    transmission shouldBe (0.92 +- 0.05)
  }

  it should "show Beer-Lambert absorption for colored glass" in {
    val coloredScene = canonicalScene.copy(
      sphere = canonicalScene.sphere.copy(absorptionCoef = 0.5)
    )
    val result = CausticsModule.fullTraceWithEnergy(100000, coloredScene)

    // Average path length through unit sphere ≈ 1.33
    // Expected transmission due to absorption: exp(-0.5 * 1.33) ≈ 0.51
    // Combined with Fresnel: 0.51 * 0.92 ≈ 0.47
    val transmission = result.depositedEnergy / result.inputEnergy
    transmission shouldBe (0.47 +- 0.10) // Higher tolerance for complex interaction
  }
}
```

### Pass Criteria
- [ ] Total energy (deposited + absorbed + reflected) within 5% of input
- [ ] Clear glass transmission ~92% (Fresnel losses at two interfaces)
- [ ] Absorption follows Beer-Lambert for colored glass

---

## Steps 6-8: Statistical and Visual Validation

> **Note:** Steps 6-8 require rendered images and statistical analysis over multiple iterations.
> These are documented here for completeness but are validated visually or with reference images.

### Step 6: PPM Convergence (C6)

**Goal:** Verify variance decreases monotonically after iteration 3.

**Validation:**
1. Render caustics at iterations 1, 2, 3, 5, 10, 20
2. Compute per-pixel variance within caustic region
3. Verify variance(N) < variance(N-1) for N > 3

### Step 7: Caustic Brightness (C7)

**Goal:** Verify caustic is visibly brighter than ambient.

**Validation:**
1. Identify caustic region (high variance area on floor)
2. Compute mean brightness in caustic vs ambient floor
3. Verify caustic_brightness > 1.5 × ambient_brightness

### Step 8: Reference Match (C8)

**Goal:** Match known-good reference image.

**Validation:**
1. Render canonical scene with fixed parameters
2. Compare against reference image using SSIM
3. Verify SSIM > 0.90

**Reference Image Creation:**
- Render with verified implementation (Steps 1-7 passing)
- Or obtain from trusted external renderer (LuxRender, PBRT, etc.)
- Store as `test/resources/caustics_reference.png`

---

## Test Execution Order

```bash
# Run tests in order, stopping at first failure
sbt "testOnly *PhotonEmissionSpec"     # Step 1
sbt "testOnly *SphereHitRateSpec"      # Step 2
sbt "testOnly *RefractionAccuracySpec" # Step 3
sbt "testOnly *FocalPointSpec"         # Step 4
sbt "testOnly *EnergyConservationSpec" # Step 5
```

## Debugging Failed Tests

### Step 1 Failures
- Check random number generator initialization
- Verify emission kernel launch dimensions
- Log sample of directions and check manually

### Step 2 Failures
- Visualize ray-sphere intersections
- Check sphere transform matrix
- Verify ray origin/direction aren't swapped

### Step 3 Failures
- Test refraction function in isolation with known values
- Check normal direction (inward vs outward)
- Verify IOR isn't inverted (1/n vs n)

### Step 4 Failures
- Plot photon hit positions to visualize pattern
- Check if centroid is offset systematically (indicates bias)
- Verify thick lens formula vs thin lens approximation

### Step 5 Failures
- Track energy through each stage separately
- Check for "lost" photons (neither deposited nor tracked)
- Verify Fresnel coefficients are symmetric

---

## Implementation Status

| Step | Test | Status |
|------|------|--------|
| 1 | PhotonEmissionSpec | ⬜ Not implemented |
| 2 | SphereHitRateSpec | ⬜ Not implemented |
| 3 | RefractionAccuracySpec | ⬜ Not implemented |
| 4 | FocalPointSpec | ⬜ Not implemented |
| 5 | EnergyConservationSpec | ⬜ Not implemented |
| 6 | ConvergenceSpec | ⬜ Not implemented |
| 7 | BrightnessSpec | ⬜ Not implemented |
| 8 | ReferenceMatchSpec | ⬜ Not implemented |

---

## References

- [arc42 Section 10 - Quality Requirements](../docs/arc42/10-quality-requirements.md)
- [arc42 Section 8 - Physics Concepts](../docs/arc42/08-crosscutting-concepts.md)
- PBRT Book, Chapter 16: Light Transport III - Bidirectional Methods
- Jensen, H.W. "Realistic Image Synthesis Using Photon Mapping"
