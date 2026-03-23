# Caustics Scene Parameter Iteration Log

Tracking all parameter choices and outcomes to prevent re-doing work after context compaction.

## PBRT Reference Parameters (Ground Truth)
- Camera: (0, 4, 8) → lookAt (0, 0, 0), up (0, 1, 0), FOV 45°
- Light: Point light at (0, 10, 0), intensity 500
- Sphere: origin (0,0,0), radius 1.0, glass IOR 1.5
- Floor: Y = -2, diffuse gray (0.8, 0.8, 0.8)
- Background: black
- Integrator: BDPT, 1024 samples, 800×600
- Reference image: `optix-jni/test-resources/caustics-references/pbrt-reference.png`

## Our Renderer Constraints
- No point light support — using directional lights
- Light direction convention: `direction` field points TOWARD light source (for N·L shading)
- Photon emitter uses `light.direction` as photon travel direction (convention mismatch — deferred fix)
- No CLI `--bg-color` flag yet (to be added in sprint)
- DSL `background = Some(Color.Black)` should set background via `OptiXEngine`

---

## Base Scene Iterations (no caustics)

### v1 (pre-compaction): Camera (0, 0.5, 10)
- Camera: (0, 0.5, 10), lookAt (0, -0.5, 0)
- Lights: two directional — (0,-1,0) i=1.0 and (0,1,0) i=1.0
- Background: Color.Black (via DSL)
- Plane: Y = -2, gray 0.8
- **Outcome**: Floor visible, sphere visible, NO shadow (forgot --shadows flag)
- **Issues**: Background appeared black ✓, but two lights created harsh split on sphere

### v2: Camera (0, 3, 8), single light
- Camera: (0, 3, 8), lookAt (0, -0.5, 0)
- Lights: single directional (0, 1, 0) intensity 1.0
- Background: Color.Black (via DSL)
- Plane: Y = -2, gray 0.8
- Caustics: None
- Shadows: enabled via CLI --shadows
- **Outcome**: Shadow faintly visible ✓, perspective closer to PBRT ✓
- **Issues**: Bottom half of sphere fully black

### v3: Camera (0, 4, 8) — exact PBRT match (REJECTED)
- Camera: (0, 4, 8), lookAt (0, 0, 0) — exact PBRT parameters
- Lights: single directional (0, 1, 0) intensity 1.0
- **Outcome**: ENTIRE image is floor — infinite plane covers all rays from this high angle
- **Root cause**: Our infinite plane differs from PBRT's finite quad. Camera too high = no horizon.
- **Decision**: Camera (0, 4, 8) does NOT work with infinite planes. Reverted.

### v3-debug: Red floor to confirm
- Same as v3 but floor color (0.8, 0.2, 0.2)
- Confirmed: entire image is red floor, no background visible

### v4: Camera (0, 0.5, 10) restored, single directional light
- Camera: (0, 0.5, 10), lookAt (0, -0.5, 0)
- Lights: single directional (0, 1, 0) intensity 1.0
- Background: Color.Black ✓ VISIBLE (black above horizon)
- Plane: Y = -2, gray 0.8
- Caustics: None
- Shadows: enabled via CLI --shadows
- **Outcome**: Horizon visible, black background. But directional light = no shadow.
- **Issues**: No shadow (glass is white, so shadow attenuation = alpha * (1-color) = 0)

### v5: Point light at (0,10,0), intensity 50
- Lights: directional (0,-1,0) i=0 (for caustics photon emitter only) + point (0,10,0) i=50
- **Outcome**: Floor gradient visible, natural falloff. Too dark overall.

### v6: Point light intensity 500
- **Outcome**: Floor overexposed (blown out white near center). Dark shadow circle below sphere visible!

### v7: Point light intensity 200
- **Outcome**: Close to reference. Floor slightly overexposed near center. Shadow ring visible.

### v8: Point light intensity 150
- **Outcome**: Good floor brightness. Sphere slightly dim. Shadow ring visible.

### v9: Point light intensity 180 (CURRENT BEST BASE)
- Camera: (0, 0.5, 10), lookAt (0, -0.5, 0)
- Lights: directional (0,-1,0) i=0.0 + point (0,10,0) i=180
- Background: Color.Black ✓
- Plane: Y = -2, gray (0.8, 0.8, 0.8)
- Caustics: None
- Shadows: enabled via CLI --shadows
- **Outcome**: Good match to PBRT reference. Floor brightness reasonable, natural gradient.
  Shadow/refraction ring visible below sphere. Sphere upper half properly lit.
- **Remaining differences from PBRT**:
  1. Camera position differs (ours: 0,0.5,10 vs PBRT: 0,4,8) due to infinite plane
  2. Floor slightly brighter/more saturated near center than PBRT
  3. Sphere top is slightly less bright than PBRT
  4. PBRT shows stronger specular highlight on sphere top

### v10: Camera (0, 2, 12) — higher + farther
- **Outcome**: More floor visible, horizon shows. Sphere too small (too far).

### v11: Camera (0, 2.5, 10) — higher, same distance
- **Outcome**: Too high — background hardly visible. Floor dominates.

### v12: Camera (0, 1.5, 10) — split difference (COMMITTED BASE)
- Camera: (0, 1.5, 10), lookAt (0, -0.5, 0)
- Lights: directional (0,-1,0) i=0.0 + point (0,10,0) i=180
- Background: Color.Black ✓
- Plane: Y = -2, gray (0.8, 0.8, 0.8)
- Caustics: None (disabled for base scene)
- Shadows: --shadows
- **Outcome**: Good compromise. Black background visible, floor below sphere exposed
  for caustic area. Sphere size reasonable. Natural point-light gradient.
- **Committed as base scene. Ready for caustics iteration.**

---

## Caustics Iterations

### caustics-v1 (pre-compaction): Camera (0, 0.5, 10), HighQuality, two lights
- Camera: (0, 0.5, 10), two directional lights (0,-1,0) and (0,1,0)
- Caustics: HighQuality (500k photons, 20 iter, alpha 0.8)
- caustic_scale: 1.0 (hardcoded in shader)
- **Outcome**: Caustic spot massively blown out (pure white ellipse)
- **Issues**: Way too bright, completely overrides shadow, needs scale reduction

### caustics-v2: Camera (0, 0.5, 10), HighQuality, single light, shadows
- Saved to /tmp/caustics-base-shadows.png
- Camera: (0, 0.5, 10), single directional (0,-1,0) + (0,1,0)
- Caustics: HighQuality (500k photons, 20 iter, alpha 0.8), 10M total photons
- caustic_scale: 1.0 (hardcoded in shader)
- **Outcome**: Caustic still massively blown out white ellipse
- **Issues**: Same as caustics-v1

### caustics-v3: scale=0.3, two directional lights
- Camera: (0, 0.5, 10), directional (0,-1,0) i=1.0 + (0,1,0) i=0.8
- caustic_scale: 0.3
- **Outcome**: Still blown out white ellipse

### caustics-v4: scale=0.01, two directional lights
- **Outcome**: No visible caustic (too dim). Also floor too dark (second light i=0.8)

### caustics-v5: scale=0.3, directional (0,-1,0) i=0.3 + directional (0,1,0) i=1.0
- **Outcome**: Still blown out. Directional light i=0.3 still produces too much caustic flux

### caustics-v6: scale=0.01, directional (0,-1,0) i=1.0 + directional (0,1,0) i=0.8
- **Outcome**: Floor too dark (0.8 intensity directional). Caustic not visible at 0.01.

### caustics-v7: scale=0.1, point light base (CURRENT SETUP)
- Camera: (0, 1.5, 10), lookAt (0, -0.5, 0)
- Lights: directional (0,-1,0) i=1.0 (photon emitter) + point (0,10,0) i=180
- Caustics: HighQuality (500k photons, 20 iter, alpha 0.8)
- caustic_scale: 0.1
- **Outcome**: Caustic visible as bright circular area. Still a bit too bright/spread.

### caustics-v8: scale=0.03
- Same as v7 but caustic_scale=0.03
- **Outcome**: More subtle caustic with concentric ring pattern. Brightness reasonable.
  Still more spread out than PBRT's tight focal spot. Good brightness level.

---

## Key Findings
- Camera (0, 4, 8) does NOT work because infinite plane covers all rays (no horizon)
- Camera (0, 1.5, 10) gives good composition (committed as base scene)
- Background Color.Black DOES work — it was hidden by the infinite plane at high camera angles
- Glass sphere shows no shadow from shadow rays: alpha*(1-color)=0 for white glass.
  Shadow in PBRT reference IS the caustic effect (light redistribution via BDPT).
- Point light at (0,10,0) i=180 gives good floor illumination matching PBRT
- Directional light (0,-1,0) at i=1.0 drives photon emitter (N·L<0 for floor, no shading effect)
- caustic_scale needs to be ~0.03–0.05 range for reasonable brightness
- Caustic pattern is too spread out vs PBRT (rings vs tight focal spot) — may need
  smaller initial radius or more PPM iterations for convergence

### caustics-v9: scale=1.0, grid-accelerated deposition (2026-03-19)
- Same scene as v7/v8 but with grid-accelerated photon deposition
- caustic_scale: 1.0 (physics-based, no arbitrary scaling)
- Grid: 60³ cells, cell_size = initial_radius (0.1), bounds (-3,3)³
- **Outcome**: Brightness 54% of PBRT reference at 10K × 3 iterations. Passes 50% test.
  Grid acceleration enables higher photon counts for future quality improvement.

### caustics-v10: Gaussian kernel + exponential tone mapping (2026-03-23)
- Same scene as v9 (grid-accelerated, scale=1.0)
- **Changes from v9:**
  1. Gaussian kernel weighting in `depositPhoton()`: sigma = radius/4, weight = exp(-d²/(2*sigma²))
  2. Exponential tone mapping: `1 - exp(-L * exposure)` with exposure=0.06 (replaced Reinhard)
  3. Screen blending: `1 - (1-base)*(1-caustic)` (unchanged from v9)
  4. Removed min radius clamp — natural PPM convergence only
- **Outcome**: Brightness ratio 38.3% of PBRT reference at 10K × 3 iterations.
  Gaussian kernel reduces effective energy (distant photons contribute less) but improves
  visual quality: smoother center-to-edge falloff, no hard cutoff at radius boundary.
  Exponential tone mapping preserves more contrast than Reinhard in the bright focal center.
- **Final tuned parameters:**
  - `caustic_exposure`: 0.06
  - Tone mapping: exponential `1 - exp(-L * exposure)`
  - Blending: screen `1 - (1-base)*(1-caustic)`
  - Gaussian sigma: radius/4 (sigma_sq = radius_sq / 16)
  - Min radius clamp: none
  - `initialRadius`: 1.0 (HighQuality preset)
  - `alpha`: 0.8 (HighQuality preset)
  - Point light: y=10, intensity=200

## Sphere-Specific Limitations

The current implementation is **heavily sphere-specific**. The refraction physics (Snell's law,
Fresnel, Beer-Lambert, TIR) is fully general, but geometry handling is hard-coded for a single sphere:

| Component | How it's sphere-specific |
|-----------|-------------------------|
| Photon tracing | Manual `intersectSphere()` — does NOT use OptiX ray tracing |
| Surface normals | `normalize(hit_point - sphere_center)` — only valid for spheres |
| Photon emission | Importance-sampled cone toward `sphere_center`/`sphere_radius` |
| IOR | Global `params.sphere_ior` — single value for one sphere |
| Absorption | Reads from global `params.sphere_color` |
| Refraction guard | `hit_sphere` boolean — assumes one refracting object |
| Grid bounds | Hard-coded +/-3.0 |

Generalizing to arbitrary geometry would require: OptiX-traced photon rays, per-instance
material lookup, generalized photon emission targeting all glass objects, dynamic grid bounds.
This would be a significant refactor (future sprint). The physics core wouldn't change.

## Next Steps
- Target 20% reference match with higher photon counts (now practical with grid)
- Consider generalizing beyond sphere-specific geometry (future sprint)
- Consider per-instance material lookup for multi-object scenes
