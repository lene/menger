# Caustics validation harness (Sprint 33)

Layered validation of menger's progressive-photon-mapping caustics against pbrt-v4.

## Layers

- **L1 analytic (CPU):** `menger-app/src/test/scala/menger/caustics/CausticsPhysicsSuite.scala`
  — Snell, exact Fresnel, focal point, emission pdf. No GPU.
- **L2 statistical (GPU):** `Caustics{Emission,Energy,Convergence,Brightness}*Suite`
  — energy conservation, convergence, brightness, hit rate via `CausticsStats`.
- **L3 converged reference (this directory):** menger render vs a committed pbrt render,
  compared in linear space.

Bytewise menger-vs-pbrt is impossible (different Monte-Carlo samplers/RNG). Bytewise is
reserved for menger-vs-menger determinism (`RenderDeterminismSuite`).

## Files

- `scenes/*.pbrt` — hand-authored pbrt-v4 twins of menger scenes.
- `render-pbrt-references.sh` — **manual/CI only.** Renders each scene, gates on a
  2×-budget convergence check, converts EXR→PFM (clipped linear) + EXR→PNG (sRGB), and
  writes a manifest. Never called by git hooks.
- `compare-caustics.sh <menger.pfm> <reference.pfm>` — MSE + FLIP vs committed reference,
  pass/fail against `thresholds.txt`. This is what the integration suite calls.
- `thresholds.txt` — per-scene MSE/FLIP bounds (locked in Task 33.8).
- Committed references live in `../reference-images/caustics/` (PFM + PNG + manifest).

## Metric choice

Primary perceptual gate is **FLIP** (Andersson et al. 2020, NVIDIA), provided by pbrt's
`imgtool` and purpose-built for renderer comparison. It replaces the originally-planned SSIM:
the installed ImageMagick 6 lacks an SSIM metric, and FLIP is the better fit here. MSE is a
coarse secondary energy check. Both run on the clipped-linear PFM pair, so the comparison is
independent of the tone mapper.

## Comparing in linear space

menger writes a linear float PFM via `--save-name foo.pfm` (values are 8-bit-quantized
linear, clipped to [0,1] — see `docs/BACKLOG.md` F-HDR-FILM for the true-HDR follow-up). Use
`--tonemap none --exposure 1` so the render is linear before quantization. The pbrt side
applies the identical `--clamp 1` on EXR→PFM, so the two are directly comparable.

## Light-unit convention

pbrt point-light `"rgb I" [500 500 500]` is **radiant intensity** in W/sr. menger's
`Light.intensity` for a point light must map to the same physical quantity for brightness to
match; the twin scenes assume `intensity = I` (W/sr) with no extra 1/4π or distance factor
baked in. This convention is the reference against which the emission-pdf fix (Task 33.3) is
judged.

## Regenerating references

```bash
# Full (800x600) + gated (400x300), high budget — slow:
bash scripts/caustics-validation/render-pbrt-references.sh canonical-caustics

# Gated resolution only, faster:
RESOLUTIONS="400x300" SPP=512 bash scripts/caustics-validation/render-pbrt-references.sh canonical-caustics
```

Commit the regenerated `../reference-images/caustics/*.{pfm,png,manifest.txt}` alongside any
change that legitimately alters the reference.
