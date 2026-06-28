# Sprint 31: L-Systems in 3D and 4D

**Sprint:** 31 - L-Systems
**Status:** Not Started
**Estimate:** ~28 hours
**Branch:** `feature/sprint-31`
**Dependencies:** Sprint 29 (curves primitive — stems render as swept curves;
cylinder chains are the fallback if curves are unavailable)
**Feature ID:** F9 in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)
(promoted from ROADMAP backlog)

---

## Goal

Lindenmayer-system fractals: string-rewriting grammar engine, 3D turtle interpretation
producing curve/tube geometry, DSL and CLI integration with classic presets, and a 4D
turtle extension that runs the same grammars through the existing 4D projection
pipeline — L-system fractals rotating in 4D are something almost no other renderer
shows.

---

## Success Criteria

- [ ] `--objects type=lsystem:preset=tree:level=5` renders a recognizable 3D tree
- [ ] Custom grammars definable in the DSL (axiom + rules + angle)
- [ ] Stochastic rules supported with a fixed seed (deterministic renders)
- [ ] A 4D L-system (e.g. 4D Hilbert curve) renders and responds to
      `rot-x-w / rot-y-w / rot-z-w` like other 4D objects
- [ ] String-rewriting engine fully unit-tested against published ABOP examples
- [ ] All tests pass

---

## Tasks

### Task 31.1: L-System Grammar Engine

**Estimate:** 6h

Pure Scala, no GPU involvement — lives in `menger-app` objects layer (promote to
`menger-common` only if optix-jni-independent reuse appears).

**Implementation:**
- `LSystemGrammar(axiom: String, rules: Map[Char, Seq[(Double, String)]])` —
  deterministic rules are the single-entry weight-1.0 case; stochastic rules select
  by weight from a seeded RNG
- `rewrite(n: Int): String` with a hard output-length guard (~10⁷ symbols → fail
  fast with a clear error instead of OOM; growth is exponential in `n`)
- Turtle alphabet (ABOP standard): `F` draw forward, `f` move forward, `+ -` yaw,
  `& ^` pitch, `\ /` roll, `|` turn around, `[ ]` push/pop state, `!` decrement
  width, `'` increment color index; unknown symbols ignored (they exist for grammar
  bookkeeping)
- Validation: rules referencing symbols not in the alphabet+rule set produce
  warnings, not errors (standard L-system practice)

**Tests:** algae (A→AB, B→A: assert Fibonacci lengths), Koch curve string equality
at n=3, branch push/pop balance checker, stochastic determinism under fixed seed.

---

### Task 31.2: 3D Turtle Interpretation

**Estimate:** 8h

**Implementation:**
- Turtle state: position `Vec3`, orthonormal frame (heading H, left L, up U) as a
  3×3 matrix, current width, color index; branch stack of states
- Rotations: `+ -` rotate about U, `& ^` about L, `\ /` about H, by the grammar's
  angle parameter (matrix multiply — no quaternions needed at this scale)
- Geometry emission: consecutive `F` segments accumulate into control-point
  polylines → one `Curve` (Sprint 29) per branch run, with per-vertex widths from
  `!` decay; fallback path emits cylinder-per-segment when curves are disabled
- Optional leaf symbol `L` → small sphere or cone at turtle position (preset-defined
  material)
- Presets (`LSystemPreset` enum): `Tree` (ABOP fig 1.24-style), `Bush`, `Fern3D`,
  `HilbertCurve3D`, `KochIsland` — each a named grammar + angle + width decay +
  default materials
- Scale normalization: after generation, fit bounding box to `size` parameter so
  presets compose with existing scene conventions

---

### Task 31.3: DSL + CLI Integration

**Estimate:** 4h

**Implementation:**
- DSL:
  ```scala
  LSystem(
    axiom = "F", rules = Map('F' -> "F[+F]F[-F]F"),
    angleDegrees = 25.7f, iterations = 4,
    segmentLength = 0.1f, widthDecay = 0.7f,
    seed = 42L, material = Material.Matte,
  )
  ```
  plus `LSystem.preset(LSystemPreset.Tree, iterations = 5)`
- CLI: `--objects type=lsystem:preset=tree:level=5:size=1.5` (presets only on the
  CLI; custom grammars are DSL-only — rule maps don't fit the `key=value` syntax)
- Geometry registry entry following the Sprint 19 pattern

---

### Task 31.4: 4D Turtle Extension

**Estimate:** 6h

**Implementation:**
- Turtle frame becomes four orthonormal `Vec4` axes (H, L, U, W); rotations are
  Rotations in the six coordinate planes — extend the alphabet with `> <` (HW-plane)
  and reuse `& ^ \ /` semantics within the 4D frame
- Generation happens in 4D coordinates; segments are projected to 3D through the
  existing 4D rotation + perspective projection pipeline (same parameters as
  tesseract: `eye-w`, `screen-w`, `rot-x-w/y-w/z-w`), then emitted as curves
- Projection note: per-segment endpooints project independently; curve control
  points are projected, not the swept surface — acceptable approximation at typical
  segment lengths, documented
- Presets: `HilbertCurve4D`, `Tree4D` (3D tree grammar with one 4D-rotation symbol
  injected per branch level)
- Animation: 4D-rotation animations re-project control points per frame; this is the
  rebuild path (no GAS refit for curves initially — note as a future optimization)

---

### Task 31.5: Tests + Documentation

**Estimate:** 4h

- Unit: grammar tests (31.1), turtle-state tests (frame stays orthonormal after long
  command sequences — accumulate 10⁴ rotations, assert determinant ≈ 1), 4D frame
  tests
- Integration: tree preset reference image, 4D Hilbert curve reference image
- `scripts/manual-test.sh`: tree + 4D Hilbert rotating (append at end)
- User guide: L-Systems section (alphabet table, presets, custom grammar example,
  4D extension); CHANGELOG.md entry

---

### Task 31.6: CODE_IMPROVEMENTS Resolution

**Estimate:** 2h

Carried-forward low-priority items from Sprint 29–30:

- **L-upload-texture-file-raw-int** (1h): Decide fail-fast vs graceful-skip semantics
  for `uploadTextureFromFile` returning raw negative `Int`. Three production callers
  treat negative as "skip and continue." Either change to throw `TextureUploadException`
  (consistent with `uploadTexture`) or document the divergence.
- **L-project4d-async-error** (1h): Document the CUDA error-handling contract in
  `project4d.cu`. `cudaGetLastError()` at line 147 only captures launch-configuration
  errors; `cudaDeviceSynchronize` lives in the caller. Either move the sync inside
  or document the caller-responsibility contract explicitly.

### Task 31.7: Architecture Backlog Items

**Estimate:** 2h (T7) + 16h (T3) = 18h

From ARCHITECTURE_BACKLOG.md, identified in Sprint 30 architecture review:

- **T7 (2h): Render determinism + JNI fault-injection tests.**
  Render canonical scene twice → assert identical PNG output. Add scalamock tests
  forcing `-1` return from instance-adding JNI calls (sphere, mesh, curve, cylinder,
  cone) and verify `Try`→`Failure` propagation. Locks down reproducibility claims.
- **T3 (16h): Native memory-leak gate for menger-geometry.**
  Restore real Valgrind + compute-sanitizer checks in pre-push hook (currently
  stubbed — return 0 unconditionally). Wire to CI on the NVIDIA GPU runner.
  Target: in-repo native code (`MengerJNIBindings.cpp`, `project4d.cu`,
  `caustics_ppm.cu`) must pass leak checks on every push. This is the
  highest-impact unguarded invariant from the Sprint 30 arch review.

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 31.1 | Grammar engine (rewriting, stochastic, validation) | 6h |
| 31.2 | 3D turtle → curve geometry + presets | 8h |
| 31.3 | DSL + CLI integration | 4h |
| 31.4 | 4D turtle extension | 6h |
| 31.5 | Tests + documentation | 4h |
| 31.6 | CODE_IMPROVEMENTS (L-upload-texture, L-project4d-async) | 2h |
| 31.7 | Architecture backlog (T7 determinism tests + T3 leak gate) | 18h |
| **Total** | | **~48h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Integration + manual test scripts cover 3D preset and 4D L-system
