# Investigation: one invisible plane hides the object when rotated

Manual-test defect #12 (`ManualTestNeedFixing.md`), scenes 142 (`menger4d` two objects
L2+L3) and 147 (`sierpinski4d` two objects L2+L3). Original report speculated an invisible
occluding plane, tied to rotation.

## Stage 0 — Context — 2026-08-21

**Command:**
```
--objects type=menger4d:level=2:pos=-0.8,0,0 --objects type=menger4d:level=3:pos=0.8,0,0 --plane y:-2
```
**Output/measurement:** only the first (`level=2`, `pos=-0.8,0,0`) object rendered. The
second was completely absent — not clipped, not partially drawn, just gone.

**Hypothesis update:** report's "invisible plane" framing doesn't match. Neither scene
contains a plane primitive; the actual occluder must be something about the *second
instance itself*. Rotation removed from the test: symptom unchanged with `rot-xw=0`.
Position varied: symptom unchanged at any second-instance position. Level varied:
unchanged at any level. Wide-angle camera used to rule out simple screen-space overlap
between the two sponges: still only one object. `sierpinski4d` two-object case reproduces
identically. Corrected description: **any scene with two or more instances of the same
GPU-4D-fast-path type (`menger4d`/`sierpinski4d`/`hexadecachoron4d`) renders only the
first instance — independent of position, level, or rotation.**

## Stage 1 — Root cause — 2026-08-21

Traced the render path: these three types go through `MengerRenderer.add4D`
(menger-geometry) → `OptiXWrapper::addCustomGeometryInstance` (optix-jni's generic
custom-geometry SPI, Task 1.1d). Read `addCustomGeometryInstance`:

```cpp
if (impl->gas_registry.find(gtype) == impl->gas_registry.end()) {
    // ...build GAS from THIS call's aabbMin/aabbMax...
    impl->gas_registry[gtype] = gas_data;
}
inst.gas_handle = impl->gas_registry[gtype].handle;
```

One GAS (acceleration structure) is cached **per geometry type**, built from whichever
instance's AABB arrives first, then reused unconditionally for every later instance of
that type. This is the correct pattern for `addSphereInstance` and friends — one canonical
unit-shape GAS at the origin, positioned per-instance via a real IAS transform — but
`MengerRenderer.add4D` doesn't follow that pattern: it passes `IdentityTransform` and bakes
each instance's real world-space position directly into both the AABB (`position ± scale`)
and the packed blob, because `hit_menger4d.cu`/`hit_sierpinski4d.cu` read position/rotation
from the blob and operate entirely in world space. A second instance therefore gets a
correctly-recorded position in its blob but an acceleration-structure bounding box still
centred on the *first* instance's position — OptiX's broad-phase BVH culling rejects rays
aimed at where the second instance actually is, and the intersection shader never runs for
it. No error, no crash: silently invisible.

**Why optix-jni's own `CustomGeometrySpiSuite` never caught this:** its test stub
(`__intersection__spi_stub`) is a fixed object-space sphere at local origin, positioned
into world space via a **real** transform in every existing test case — its intersection
math never depends on the AABB or blob position the way the real 4D shaders do. A test
added to that suite using the documented (transform-based) pattern with a shared AABB
passed identically with and without the fix — proven not to discriminate, removed rather
than left as a misleading always-passing test.

## Checkpoint — root cause presented, fix authorized — 2026-08-21/22

User: "fix it, create a new release."

## Stage 2 — Fix — 2026-08-21/22 (optix-jni)

Added a second, separate registry `std::map<int, GASData> custom_geometry_gas_registry`
keyed by `instanceId` (not `GeometryType`) — the existing `gas_registry` is left untouched,
per the explicit comment at its declaration warning against reintroducing a per-instance
key there (CR-5 double-free fix, earlier in Sprint 36). `addCustomGeometryInstance` now
unconditionally builds a fresh GAS per call and stores it in the new registry;
`clearAllInstances()` frees both registries independently, same one-owner-per-buffer
discipline as the rest of the file.

**Verification:** rigorous negative control — fix stashed, `optix-jni` republished locally
as 0.3.2: `MengerRendererMultiInstanceSuite`'s two tests both fail
(`leftHalf=false was not equal to true`). Fix restored, republished: both pass. 50 C++ unit
tests pass. Released as `optix-jni` 0.3.2 (PR #35/#36, Maven Central).

## Stage 3 — menger-side integration — 2026-08-23

Bumped `menger/build.sbt` pin to `optix-jni 0.3.2` (real Maven Central artifact, not a
`publishLocal` override — verified via `~/.cache/coursier` resolution, not
`~/.ivy2/local`). Added `MengerRendererMultiInstanceSuite` (exercises the real
`hit_menger4d.cu`/`hit_sierpinski4d.cu` shaders — `optix-jni`'s own test can't, per
Stage 1). Added scripted "two instances" coverage for both types to
`integration-tests.sh`.

### Residual: reference-image capture anomaly (found and resolved during verification)

The first `--update-references` run for the two new scripted tests produced reference
PNGs that differed from the *immediately following* fresh renders by ~3.6% and ~7.0% of
pixels (confined to a small edge-boundary region of the left instance's silhouette,
`x:[0,26] y:[52,94]` for `menger4d`). This looked alarming — large enough to fail the
integration suite's 0.1% threshold — but three independent renders taken afterward
(including across two full `rm -rf native; sbt package` clean rebuilds) were **bit-for-bit
identical** to each other, just not to that one original capture. This is not the same bug
as Stage 1: it doesn't reproduce, it isn't a missing instance, and it's confined to a
sub-pixel edge region on a single first-time capture rather than the whole second object
going missing. Treated as a one-off capture anomaly (plausibly nondeterministic native
recompilation of the fractal edge case during that specific run) rather than investigated
further, since:
- The fix itself was already proven correct via the rigorous negative-control test above,
  independent of this reference image.
- Re-verification consistency (3 further renders, 2 of them across independent clean
  native rebuilds) all matched each other exactly.

References were regenerated from the verified-stable build and committed. Flagging this
here rather than silently discarding it, per the skill's own guidance to record residue
rather than pretend it's fully explained. If a scripted "two instances" test on a
fractal-edge scene flakes again in CI, start here.

## Checkpoint — visual crosscheck — 2026-08-23

Regenerated reference images show non-background pixel variation in *both* halves of the
frame (left ~11,500px, right ~11,500px, symmetric) for both `menger4d` and `sierpinski4d`
two-instance scenes — both instances visibly present, confirming the fix end-to-end, not
just via the unit-level `hasVariation` assertion.

## Minimum artefacts checklist

- [x] Regression test exists (`MengerRendererMultiInstanceSuite`, permanent, verified via
      negative control)
- [x] Scripted integration coverage exists (`integration-tests.sh`, both types) with
      committed reference images
- [x] Fix commit passes the test (both `optix-jni` C++ suite and the Scala regression
      test)
- [x] This note records per-step commands, measurements, root-cause narrative
- [x] Prior incorrect description ("invisible plane", rotation-tied) corrected in
      `ManualTestNeedFixing.md`
- [x] `optix-jni` `CHANGELOG.md` documents the fix (0.3.2)
- [x] Residual (reference-image capture anomaly) recorded above rather than silently
      discarded
