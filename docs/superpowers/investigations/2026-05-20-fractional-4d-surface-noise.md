# Investigation: Fractional 4D surface noise

**Affected types:** menger4d, sierpinski4d, hexadecachoron4d (all fractional levels)
**Reported:** 2026-05-20

---

## Stage 0 — Reproduce — 2026-05-20

**Command:**
```
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a ./menger-app-0.6.2/bin/menger-app \
  --headless -s /tmp/debug_frac_menger4d.png --objects type=menger4d:level=2.5
```

**Output/measurement:** Render succeeded (exit 0). Image saved to `/tmp/debug_frac_menger4d.png`.

**Observed defect (cause-neutral):**  
The fractal surface shows random per-pixel variation in intensity and transparency with no spatial coherence — adjacent pixels frequently differ strongly in brightness, creating a dense speckled/salt-and-pepper appearance across the entire fractal geometry and the background visible through it.

**Hypothesis update:** Description is cause-neutral. Proceeding to Checkpoint 1 (user agreement on wording).

---

## Stage 1 — Detector — 2026-05-20

**Invariant:** Mean absolute horizontal gradient across crop (220,180,560,420).
Smooth/correct surface → low (~1.3–1.7). Z-fight noise → high (≥3.5).

**Command (metric computation):**
```python
img = np.array(Image.open(path).crop((220,180,560,420)).convert("L"), dtype=float)
metric = np.abs(img[:, 1:] - img[:, :-1]).mean()
```

**Results across all six fixtures:**

| Fixture | Metric | Expected |
|---------|--------|----------|
| menger4d level=2 integer   | 1.313 | low ✓ |
| menger4d level=3 integer   | 1.738 | low ✓ |
| tesseract-sponge level=2   | 1.311 | low ✓ |
| tesseract-sponge-2 level=2 | 1.299 | low ✓ |
| menger4d level=1.5 frac    | 3.502 | high ✓ |
| menger4d level=2.5 frac    | 4.848 | high ✓ |

**Threshold:** 2.5 cleanly separates all fixtures. TesseractSponge/2 (complex geometry, no z-fighting) cluster with integer renders — detector measures z-fighting artifact specifically, not rendering complexity.

**Visual crosscheck:** Gradient heatmap (8× amplified) shows hotspots concentrated on fractal body surface only — same region where speckles are visible in original render.

**Hypothesis update:** Detector validated on 4 known-good and 2 known-bad fixtures. Prime suspect: two IFS instances at identical position+scale in scene builder.

**Checkpoint 2 passed** (user confirmed detector trustworthy).

---
