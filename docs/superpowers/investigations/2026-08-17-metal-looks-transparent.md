# Metal appears transparent — investigation

Sprint 36 Phase H3.2. `ManualTestNeedFixing.md` §7: DSL MengerShowcase, level-2 gold sponge
(84) · DSL PulsingSponge t=2 (92). Counterexample (renders correctly): DSL ComplexLighting (87).

## Stage 0 — Context

**Command:**
```
BIN=./menger-app-0.8.11/bin/menger-app
xvfb-run -a "$BIN" -o --headless -s 84_MengerShowcase.png --scene examples.dsl.MengerShowcase
xvfb-run -a "$BIN" -o --headless -s 92_PulsingSponge_t2.png --scene examples.dsl.PulsingSponge --t 2
xvfb-run -a "$BIN" -o --headless -s 87_ComplexLighting.png --scene examples.dsl.ComplexLighting
```

**Output/measurement:** three renders,
`/tmp/.../scratchpad/h32/{84_MengerShowcase,92_PulsingSponge_t2,87_ComplexLighting}.png`
(plus cropped/upscaled `84_crop.png`, `87_crop.png` for detail).

**Scene definitions checked** (`examples/dsl/MengerShowcase.scala`,
`PulsingSponge.scala` not yet read in full but shares the gold-material pattern,
`ComplexLighting.scala`): none of the three set an explicit `background` — all use the native
default (`OptiXConstants::DEFAULT_BG_R/G/B` = `(0.3, 0.1, 0.2)`, a dark maroon/plum).
`MengerShowcase` uses `Material.Gold` (`color=(1.0, 0.84, 0.0)`, `metallic=1.0`).

**Visual observation (v1, cause-neutral):**
- 84/92 (gold sponge): the sponge's exterior faces render a dull olive-yellow with mild
  variation. The walls of the fractal's interior square holes (the recessed cavity faces)
  render as a flat, uniform dark reddish-brown, with no visible texture or reflection detail
  — a sharp contrast against the exterior. Zoomed crop (`84_crop.png`) shows faint straight-line
  edges from further-back internal geometry overlaid on the flat dark-red fill, which reads
  visually like looking through a tinted, translucent surface rather than at an opaque one.
- 87 (three metallic spheres, counterexample): each sphere's upper hemisphere (no floor
  beneath it to reflect) is also a flat, uniform dark color, with a sharp horizontal boundary
  against the lower hemisphere, which clearly reflects the checkerboard floor in detail.
  Cropped (`87_crop.png`): the gold sphere's flat cap is a dark reddish-brown; the neighboring
  purple sphere's cap is a dark plum. Both are close to `metal_color * (0.3, 0.1, 0.2)` —
  consistent with a correctly-reflected plain background, tinted by each sphere's own metal
  color. This scene is reported as rendering correctly, i.e. this flat cap is expected/correct
  when there is genuinely nothing above to reflect.

**Comparison, not yet a claimed cause:** the color and flatness of 84/92's cavity-wall defect
visually matches 87's *correct* flat-background-reflection cap. The difference is *where* it
appears: 87's cap has a legitimate line of sight to empty background; 84/92's cavity walls
face other nearby sponge surfaces (the cavity's own opposite walls), which should be visible
via inter-reflection, not a flat background-tinted fill.

**Not yet confirmed with user** — sent crops for description sign-off before proceeding
(Checkpoint 1).

**User feedback (round 1):** neither 84/92 nor 87 shows the effect clearly. Best seen when
the horizon (plane's vanishing line) intersects the object.

**Command (round 2 — horizon-crossing repro):**
```
xvfb-run -a ./menger-app-0.8.11/bin/menger-app -o --headless -s horizon_try1.png \
  --objects type=sponge-volume:pos=0,0,0:level=2:material=gold:size=2.5 \
  --plane y:-2 --camera-pos 5,0,5 --camera-lookat 0,0,0
```
Camera at plane height y=0 (2 units above the y=-2 floor) puts the floor's vanishing line
(the "horizon") at screen mid-height, crossing straight through the gold sponge's silhouette.

**Output:** `/tmp/.../scratchpad/h32/horizon_try1.png` (+ `horizon_try1_crop.png`, 2x zoom).

**Visual observation (v2, cause-neutral):** the horizon line splits the sponge into two
starkly different regions, both exterior faces and cavity interiors:
- Below the horizon: the sponge surface (exterior faces and interior cavity walls alike)
  shows a detailed yellow-gold reflection of the checkerboard floor, perspective-correct,
  matching the floor's own checker pattern visible outside the sponge's silhouette.
- Above the horizon: the same surfaces (exterior and cavity) go flat, uniform rust-red, with
  only faint high-frequency striping artifacts in the cavities — no reflection of anything,
  not the purple background, not nearby cavity walls.
- The transition at the horizon line is a hard cutoff, not a gradient.

This is a materially different symptom description than round 1: it isn't "cavity walls vs.
sky caps," it's "reflection detail present below the camera-height horizon line, absent above
it, on the same surfaces." Round 1's images were consistent with this but didn't isolate it.

**User feedback (round 2) — CONFIRMED, Checkpoint 1 passed**, with a correction that changes
the description: the checkerboard lines below the horizon run **straight through** the object
silhouette, same phase/orientation as the floor outside it. That rules out reflection entirely
(a mirror would bend the lines at the surface) and means the defect is **transmission**, not
"flat color" — the surface is literally being seen through to whatever geometry lies behind it
along the *original, undeviated* view ray. Below the horizon that's the floor (so the checker
shows through unbent); above the horizon there is nothing behind it but background, hence flat
fill. This unifies round 1 and round 2's observations under one mechanism and matches the
bug's original name ("metal looks transparent") literally, not metaphorically.

**Final agreed description (Checkpoint 1):** a metallic surface, under some as-yet-unknown
condition, is shaded by continuing the incoming ray unchanged (transmission) instead of
reflecting it — visible wherever the transmitted path hits detailed geometry (the floor
checker) rather than uniform background.

## Stage 1 — Detector

**Attempt 1 — silhouette-vs-control pixel/edge statistics.** Rendered a scene-with-object and
a scene-with-object-removed-far-away (`goodsphere`/`goodsphere_control`, single gold sphere,
one bounce, known-good; `horizon_try1`/`horizon_control_noobj`, known-bad) and compared
pixels inside the object's silhouette against the same pixels in the control, three ways:
mean absolute pixel diff, Sobel-edge-sign phase-match, and gaussian-highpass Pearson
correlation. None discriminated — both fixtures scored similarly under every metric. Visual
inspection of a cropped overlay (`horizon_overlay.png`) still showed the checker pattern
passing through the sponge in exact phase with the control, confirming the effect is real;
the detector code just wasn't isolating it. Root cause of the failed metrics: the sponge
silhouette mixes many local tint/shading factors (per-facet lighting, AO, multiple depths),
which swamp a global pixel-level statistic even when the local phase-lock is visually obvious.

**Attempt 2 — purpose-built mirror-corridor fixture (per user direction).** Tried a two-plane
mirror corridor and an angled single mirror with two marker spheres (one along the true
reflection direction, one along the straight-through/transmission direction), meant to force
high bounce depth cheaply and give an unambiguous either/or visual answer. Both attempts had
geometry-design bugs (parallel-ray coincidences bypassing the mirror entirely, degenerate
head-on mirror facing returning pure background either way) that made the fixture
uninformative before it ever exercised the depth-cutoff path. Not pursued further — see pivot
below.

**Pivot: read the shader code directly** rather than keep iterating on empirical fixtures.
This is Stage 2 localization work, done early because Stage 1's black-box approach stalled;
the numeric-detector requirement is deferred until a concrete mechanism is confirmed, at which
point a fixture can be built *for* that mechanism instead of guessing blind.

## Stage 2 — Localization (prime suspect, not yet confirmed with user)

`handleMetallicOpaque` (`optix-jni/src/main/native/shaders/helpers.cu:1411-1452`) and
`traceReflectedRay`/`traceFinalNonRecursiveRay` (`:1250-1291`, `:1004-1036`) all compute the
reflection direction with the standard formula (`R = I - 2·dot(I,N)·N`) — reading the math in
isolation, nothing here does literal pass-through. But `traceFinalNonRecursiveRay`'s own
comment says the quiet part out loud: *"Avoids black artifacts from depth cutoff by tracing
one more reflection"* (`:997-998`) — the depth-exhausted bailout path itself issues **one more**
`optixTrace` call (`:1020-1031`) beyond the app's own `depth >= max_ray_depth` cutoff
(`:1421`).

That extra trace call is significant because of a second, separate constant reuse:
`MAX_TRACE_DEPTH = 5` (`optix-jni/src/main/native/include/OptiXData.h:14`) is used for *two*
different things — it's both the default/ceiling for the app-level `max_ray_depth` bounce
counter (`RenderConfig.h:129`, and the CLI's `--max-ray-depth` 1–5 range), **and** it is fed
directly to OptiX's own hardware pipeline recursion limit,
`pipeline_link_options.maxTraceDepth` (`PipelineManager.cpp:272-273`, `OptiXContext.cpp:573,
582`). OptiX's own recursion counter and the app's manual `depth` payload increment together
1:1 (each `traceReflectedRay` call is one nested `optixTrace`, `next_depth = depth + 1`,
`:1274`), so the app's `depth` value at a given closest-hit invocation *is* that invocation's
OptiX recursion level.

Working through the call sequence: normal bounces continue for `depth` = 0 .. `max_ray_depth
- 1`, each issuing one more nested trace, reaching OptiX recursion level `max_ray_depth`. At
that level, the closest-hit sees `depth >= max_ray_depth` and calls
`traceFinalNonRecursiveRay`, which issues **one further** nested trace — OptiX recursion level
`max_ray_depth + 1`. With `max_ray_depth` at its default/max of 5, the actual recursion reached
is 6, one level past what `pipeline_link_options.maxTraceDepth = 5` (`OptiXData.h`'s same
constant) permits. Exceeding OptiX's configured `maxTraceDepth` is undefined behavior at the
API level — depending on driver/validation settings this can mean the offending `optixTrace`
call is a no-op that leaves the payload registers holding whatever was already in them,
which would explain both symptoms observed: a flat, uniform fill (uninitialized/leftover
register state reads as one constant color) where nothing else is behind the surface, and — in
the horizon-crossing repro, if the leftover register state happens to be inherited from a
sibling/parent trace that legitimately sampled the floor — content that lines up with what's
actually behind the object.

**Attempted confirmation of the recursion-overflow hypothesis — REFUTED.** Enabling OptiX's
own overflow detection would require adding an exception program and rebuilding the pipeline
(no `MENGER_OPTIX_VALIDATION`/debug-level mechanism is actually wired up despite the comment
at `OptiXContext.cpp:160-164` suggesting one — checked, it's a no-op today). Instead, re-ran
the existing `horizon_try1` sponge repro at `--max-ray-depth` 2, 3, 4, 5 and read
`params.stats->max_depth_reached` from each run's log line:

| `--max-ray-depth` | logged `depth=1-N` | native trace calls to reach it | exceeds pipeline's `maxTraceDepth=5`? |
|---|---|---|---|
| 2 | 1-3 | 4 | no |
| 3 | 1-4 | 5 | no (exactly at the limit) |
| 4 | 1-5 | 6 | **yes** |
| 5 (default) | 1-6 | 7 | **yes** |

If the defect were caused by exceeding OptiX's compiled recursion limit, it should disappear
or look different at `--max-ray-depth 2` (comfortably inside budget). It did not —
`horizon_depth2.png` shows the identical artifact (checker passes through the sponge, flat
red above the horizon), pixel-for-pixel the same pattern as the default-depth render. **This
rules out the recursion-overflow theory**: the bug reproduces at a depth setting where no
native overflow is possible, so it isn't about exceeding OptiX's hardware/pipeline limit at
all — it fires every time `traceFinalNonRecursiveRay` runs, at any depth.

**New prime suspect, found by that negative result: a sign bug in
`traceFinalNonRecursiveRay` itself (`helpers.cu:1004-1036`).** Comparing it against the
known-correct `traceReflectedRay` (`:1250-1266`):

```cuda
// traceReflectedRay (correct):
const float dot_in = dot(ray_direction, normal);      // signed; negative for a front-facing hit
const float3 reflect_dir = ray_direction - 2.0f * dot_in * normal;

// traceFinalNonRecursiveRay (buggy):
const float cos_theta = fabsf(dot(ray_direction, normal));  // always positive
const float3 reflect_dir = ray_direction - REFLECTION_SCALE * cos_theta * normal;  // SCALE=2.0
```

Taking the absolute value flips the sign of the correction term whenever the raw dot product
is negative — which is the normal case for a front-facing hit. Decomposing
`ray_direction = I_perp + cN` (tangential + normal components, `c = dot(I,N) < 0` for a front
hit): the correct formula gives `R = I_perp - cN` (tangential unchanged, normal component
inverted — an actual mirror reflection). The buggy formula gives
`R = I_perp - 2|c|N = I_perp + 2cN`, which for a **near head-on hit** (`I_perp ≈ 0`,
`I ≈ cN`) reduces to `R ≈ 3cN ∝ I` — **the same direction as the incoming ray**, i.e. the ray
continues forward essentially unreflected. That is a literal pass-through, algebraically
derived from the formula itself, no fixture required — and it matches the observed symptom
exactly (checker pattern continuing in phase, strongest where the surface is viewed close to
head-on, which is exactly the geometry at a horizon-crossing sponge face and at a sphere's
sky-facing cap). For oblique hits the formula is still wrong (it inverts the wrong sign on
the normal component, tripling it in the incoming direction rather than flipping it), just
less visually identity-like.

**Confirmed with user (Checkpoint 3):** proceed to Stage 3 on the sign-bug fix.

## Stage 3 — Fix, and a second bug it unmasked

**Fix 1 (sign bug):** `optix-jni` `helpers.cu` — changed `traceFinalNonRecursiveRay` to use
the signed `dot(...)`, matching `traceReflectedRay`. Verified with a full native rebuild
(`sbt nativeTest`, real CUDA/OptiX, all 44 pre-existing C++ tests green) and a re-render of
`horizon_try1` under the fixed binary: **crashed** — `CUDA call 'cudaDeviceSynchronize()'
failed: an illegal memory access was encountered (700)`. Simple single-bounce scenes still
rendered fine; only the deep-recursion sponge scene crashed.

**Fix 2 (unmasked recursion bug):** the sign bug's near-identity "reflection" almost never
lands on more geometry that needs tracing, so it almost never recursed. With the sign fixed,
the bailout ray genuinely reflects and can hit another reflective/refractive surface — whose
closest-hit sees the *same* pinned depth and calls `traceFinalNonRecursiveRay` again, unbounded,
chaining real `optixTrace` calls past OptiX's compiled `maxTraceDepth`. Fixed by tagging the
bailout ray's result depth as `max_ray_depth + 1` and having every call site that used
`depth >= max_ray_depth` for this cutoff terminate immediately (plain diffuse shading, no
further trace) when `depth > max_ray_depth` — only `depth == max_ray_depth` gets the one
allowed bailout bounce. Applied at all five call sites: `helpers.cu` (`handleMetallicOpaque`),
`hit_sphere.cu`, `hit_triangle.cu`, `hit_cone.cu`, `hit_curve.cu` (`hit_triangle.cu`'s other
two depth-cutoffs already used the safe `handleFullyOpaque` pattern — only its main refraction
path needed the same fix as the others).

**Fix 3 (root config problem, per user's standing question about why 5 is so low):**
`MAX_TRACE_DEPTH = 5` (`OptiXData.h`) was being reused for two unrelated things — the
app-level `--max-ray-depth` bounce budget *and* OptiX's own compiled `maxTraceDepth` — sized
originally for glass entry/exit bookkeeping (`git log -S`: comment *"Allow internal
reflections in glass (entry + exit + reflections)"*, `optix-jni@4a1250f`), not for general
reflective-scene depth. The bailout's extra trace call could exceed that shared ceiling even
without the chaining bug. Decoupled: `MAX_TRACE_DEPTH` stays the app-level default/ceiling;
new `PIPELINE_MAX_TRACE_DEPTH = 10` sizes the actual OptiX pipeline and stack
(`PipelineManager.cpp`, `OptiXContext.cpp`), with headroom for the bailout bounce and for
raising the bounce budget later without another pipeline rebuild.

**Verification, round 1 — did the crash go away, and does the fix look right?** Rebuilt
native, republished locally (temporary version `0.3.1-h32-local-test` + a `mavenLocal`
resolver in `menger/build.sbt`, since coursier doesn't check `~/.m2` by default and had been
silently re-fetching the real, unfixed `0.3.0` from Maven Central instead of the local publish
— this cost a full debug cycle before being caught). Crash gone (confirmed on 3 repeat runs).
Diffed `horizon_try1` before/after: only **~1.1% of pixels changed** (`diff_mask.png`),
concentrated exactly in the cavity recesses — the broad "checker shows through" area outside
the holes is **pixel-identical**, before and after.

**This reframes the investigation.** The dominant visual effect that drove round 2 (checker
lines continuing unbent across most of the sponge's exterior, motivating the "transmission"
description) is very likely **not** this bug: axis-aligned Menger-sponge facets reflecting a
repeating floor grid, viewed near the horizon, can look nearly unbent under *correct* physics
— a purely horizontal mirror flips only the vertical component of the view ray, and a
repeating grid pattern barely changes appearance under that flip, especially at the specific
symmetric camera angle used for this fixture. Re-rendered the original round-1 scenes to
check against the symptom that actually motivated H3.2:

- **84 (MengerShowcase):** ~3.5% of pixels changed, concentrated in the cavity holes *and* the
  sky-facing top face (`84_diff_mask.png`). Before/after crop (`84_before_after.png`) shows
  real added detail — visible orange highlight/reflection variation inside several cavities
  that were flat before. This matches round 1's original description directly.
- **87 (counterexample):** pixel-identical before/after (max diff 0) — no regression, as
  expected (this scene never reaches the depth cutoff).
- Glass cone sanity render: still refracts correctly (H1.4 unaffected).

So: real, verified fix (crash + sign bug), with a genuine but **narrower** visual effect than
round 2's repro suggested — it explains round 1's cavity-detail complaint, not round 2's
horizon-crossing checker pattern, which is now suspected to be correct rendering, not a bug.
**Not settled — H3.2 stays open per user's explicit instruction ("not ready to close").**

**Verification, round 2 — full integration suite.** `PARALLEL_MODE=false
./scripts/integration-tests.sh` against the fixed build: **210/234 passed, 24 mismatches**
(`integration-h32.log`), all concentrated in reflection/refraction/glass/metal-heavy scenes
(caustics, tesseract/sponge-with-material, cone glass, plane+sphere, 600-/120-cell edges,
diamond dispersion, several DSL scenes, two animation freeze-frames) — no crashes, no
structural breaks in unrelated categories (L-systems, area lights, IBL, fog, video output all
clean). Consistent with genuine behavior change in the fixed code paths.

**Caution, not yet resolved:** spot-checking `cone glass` (one of the 24) found it rendered
correctly on a same-command retry — the first render was a transient GPU/CUDA glitch (residual
from the earlier crash-testing in this same session), not a real regression. **The 24-item
list has not been individually re-verified and must not be treated as a final, trustworthy
diff set** — some fraction may be similarly transient. Reference-image regeneration is
deliberately **not** done yet, pending that verification (regenerating references from a flaky
render would bake a bad reference in).

**Source-level regression test, added directly to `optix-jni`** (per user request — the fix
lives there, but until now only `menger`-level render tests exercised it). Extracted the
reflection formula into a single shared `reflect()` helper (`VectorMath.h`, `__host__
__device__`, host-C++-testable without CUDA or an OptiX context — the duplication between
`traceReflectedRay` and `traceFinalNonRecursiveRay` was itself the hazard that let the sign
typo diverge unnoticed). New `tests/VectorMathTest.cpp`, 6 gtest cases: head-on, 45°, grazing,
length-preservation, involution, and a named regression test (`HeadOnHitDoesNotDegenerateToPassThrough`)
that reconstructs the historical buggy formula and asserts the correct one no longer matches
it. All 50 native tests (44 pre-existing + 6 new) pass.

**Committed** in `optix-jni` on `feat/sprint-36`: `c1803fd` — fix + refactor + regression
test. Not pushed. Reverted the temporary local-test version/resolver hack in both repos'
`build.sbt` back to the real `0.3.0` pin; `menger`'s local checkout no longer has the fix
until a real `optix-jni` release (Maven Central is immutable — the published `0.3.0` predates
this fix, so a genuine version bump + release is required before `menger`'s pin can follow it
for real). That release step has not been taken this session.

**Re-verification of the 24 mismatches (this round).** Ran the full suite a second time,
sequential, same fixed build: **byte-identical** failure list and diff percentages both runs
— all 24 are deterministic, not flaky (the earlier `cone glass` "blank" render was an artifact
of an ad-hoc manual test at the *wrong resolution*, not the suite itself; see below).

Spot-checked three at the **correct** per-test resolution (my first manual comparisons had
used the CLI default 800×600 instead of each test's actual `--width`/`--height` — `run_test`
defaults to 200×150, `run_test_hd` to 400×300, `run_test_hires` to 800×600 — which had
partly invalidated the earlier before/after crops):
- **cone glass** (200×150, `cone_glass_correct_compare.png`): reference shows the glass cone
  as **flat opaque gray, no refraction at all**; fixed render shows correct maroon-tinted
  glass with faceted refraction and checker-through-glass distortion. Unambiguous — the old
  reference was wrong, not just different.
- **600-cell with edges** (`600cell_before_after.png`): fixed render shows more differentiated
  internal reflection detail (bright highlights) inside the polytope where the old reference
  was a more uniform gold fill.
- **icosahedron caustics** (`icosa_compare.png`): fixed render shows additional caustic
  highlight detail near the base facets that the old reference lacked.

All three are quality improvements, none are regressions. Regenerated all 24 references
(`./scripts/integration-tests.sh ... --update-references --filter "<name>"` for each, one
invocation, OR-matched). **Full suite re-run clean: 234/234.**

**Sequencing problem, not yet resolved:** these 24 reference images are currently only
correct against the *fixed* `optix-jni` binary — which today only exists via the temporary
local-test publish (`0.3.1-h32-local-test`) reverted earlier. `menger`'s `build.sbt` pin is
back to the real, unfixed, Maven-Central `0.3.0`. **Committing these reference-image changes
to `menger` now would break its own CI**, which builds against the real pin and would
reproduce the original (pre-fix) renders — a guaranteed 24-test regression the moment this
lands, until `optix-jni` actually releases the fix and `menger`'s pin is bumped to follow it.
Not committed yet — needs a decision on sequencing (hold the reference changes until the real
release + pin bump, or commit now with an explicit acknowledged CI-red window).

**Outstanding, not yet done:**
- Sequencing decision above (reference images regenerated locally, not committed to `menger`).
- Re-examine whether round 2's "horizon-crossing checker pass-through" is actually correct
  rendering (as currently suspected) or a distinct, still-unfound bug — this is why H3.2
  remains open.
- A real `optix-jni` version release before `menger` can pin the fix for real use.
- Push the `optix-jni` commit (not yet requested).
