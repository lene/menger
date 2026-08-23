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

---

## Round 3 — Stage 1 detector: occlusion test — 2026-08-18

Both reference-image sequencing items above were since resolved (committed `7089727a` in
`menger`, `38fa5d1` in `optix-jni`, CI-red window explicitly accepted). This round returns to
the one genuinely open question: is "metal looks transparent" a real defect?

### Scene changes to make the symptom unambiguous

Scenes 84 (`MengerShowcase`) and 92 (`PulsingSponge`) had a *solid* floor, so the pass-through
claim could not be judged. Changed both to a checkered floor (`Plane.checkered`) and gave the
sponge an off-axis rotation (`rotation = Vec3(25f, 15f, 10f)`) so its edges are not parallel to
the checker grid — camera rotation alone cannot do this, and axis-parallel geometry is exactly
the case where reflection and transmission are hardest to tell apart.

Renders `84_rotated.png` / `92_rotated.png` show the checker pattern apparently continuing
through the gold surface. **That appearance was, on measurement, misleading — see below.**

### The detector, and its fixture validation

**Invariant:** an opaque object must fully occlude an object placed behind it. Put a matte red
sphere (`#ff0000`) directly behind the subject, on the camera→subject axis, and count pure-red
pixels (`R>90, G<30, B<30`) inside the subject's silhouette. Transmission → red survives;
opacity → zero red.

**Command:**
```
menger-app -o --camera-pos 4,3,6 --camera-lookat 0,0,0 \
  --plane y:-3 --plane-color c0c0c0:404040 --width 600 --height 450 \
  --objects type=cube:size=2.5:material=<M> \
  --objects type=sphere:pos=-2.2,-1.7,-3.3:size=1.5:material=matte:color=#ff0000
```

**Fixture validation (required before trusting it):**

| Fixture | Expected | Measured |
|---|---|---|
| red sphere alone, nothing in front | red visible | **4212 px** |
| matte cube in front (known-good opaque) | zero red | **0 px** |
| gold sponge, no sphere in scene (false-positive check) | zero red | **0 px** |
| matte sponge, no sphere in scene (false-positive check) | zero red | **0 px** |

Detector is trustworthy: it fires on the known-bad case, is silent on the known-good case, and
does not false-positive on gold's dark shading (an earlier looser threshold did — it counted
166 "red" px on a scene containing no red object at all, and was tightened).

### Result — no transmission, in any material

| Cube material | red still visible | % of sphere |
|---|---|---|
| matte | 0 | 0.0% |
| plastic | 0 | 0.0% |
| metal | 0 | 0.0% |
| gold | 0 | 0.0% |
| chrome | 0 | 0.0% |

Gold *sponge*: 26 px of 1880 (1.4%), consistent with antialiasing at hole edges, versus 0 for a
solid cube.

**Hypothesis update: the transmission hypothesis is REFUTED.** Nothing is passing through.

### What the symptom actually is

Two follow-ups explain the appearance completely:

1. **Background colour is (76, 25, 51)** — dark maroon — sampled from the corners of a no-plane
   render. Faces that looked like flat, unshaded "holes" (the chrome cube's top face, the
   chrome octahedron's upper faces) measure **(68, 22, 45)** = background × chrome tint. They
   are correctly reflecting the sky. A chrome octahedron's lower-right face, which faces the
   floor, shows a properly foreshortened, tilted, high-frequency checker reflection — i.e.
   reflection geometry is demonstrably working on obliquely-oriented faces.

2. **A vertical mirror over an infinite horizontal floor maps that floor onto itself.**
   Reflection across the plane `x = c` sends `(x,y,z) → (2c-x, y, z)`; the floor plane `y = -3`
   is invariant. So the reflected floor *is* the same physical plane, at the same height, in the
   same perspective — the checker lines continue unbent, with no seam. This is real optics, not
   a bug.

**This invalidates the Checkpoint-1 reasoning.** The agreed description rested on "with
reflection, the checkerboard lines would change direction when meeting at the object surface."
That inference does not hold for a vertical mirror face over a horizontal floor, which is
precisely the geometry in scenes 84/92. The rotation added this round tilts the *sponge*, but
each individual sponge facet remains axis-aligned within the rotated body, so many facets stay
effectively vertical.

**Hypothesis update:** the leading explanation is now that H3.2 is **not a defect** — metallic
= 1.0 (gold, chrome) with almost no diffuse term, over an infinite self-similar checkered floor
and against a dark background, is genuinely near-indistinguishable from transparency. Needs a
user checkpoint before being closed as such.

**Residual anomaly — resolved, it is a shadow.** In `x_chrome_cube.png` a dark blob appears near
the bottom of the frame. Differencing that render against the identical scene without the sphere
(`w_chrome_nosphere.png`) isolates exactly which pixels the sphere is responsible for:

**Command:** render chrome cube with and without the red sphere; mask where `|Δ| > 25`.
**Output/measurement:** 10778 pixels changed, all inside bbox (163–371, 341–449) — floor, below
and in front of the cube, disjoint from the sphere's own screen position (216–389, 138–221).
Mean colour there goes (92, 92, 92) → (38.8, 38.8, 38.8); the two checker shades map 121 → 57 and
40 → 19, both scaled by the same ≈0.47 factor.
**Hypothesis update:** uniform multiplicative darkening of both checker shades is a cast shadow,
not geometry leaking through. The sphere is fully occluded (0 pure-red px); only its shadow
reaches visible floor. No anomaly remains — this closes the last open thread from round 3.

Artefacts: `e_A_gold` … `e_H_matte_redfloor`, `r_*` (marker series), `x_*` (occlusion series),
`z_*_chrome` (tilted-face series), all under the session scratchpad.

**Note on a failed experiment:** `--rot-x` / `--rot-y` are no-ops for CLI `--objects` geometry —
`y_chrome_roty30.png` and `y_chrome_rotxy.png` are pixel-identical to the unrotated cube. Object
rotation is only reachable via the DSL `rotation` parameter. Do not re-run that test.
(Also: a CLI render with no `--objects` at all fails with *"SceneConfig must provide
objectSpecs"* — for a floor-only reference frame, add a tiny sphere far off-camera.)

## Round 3b — correlation test, and a correction to my own optics argument

The seamlessness argument above ("a vertical mirror maps the horizontal floor onto itself, so
the checker must continue unbent") is **wrong as stated**, and the rotated-cube renders disprove
it: a cube yawed 30° about Y still has perfectly vertical faces, yet its reflections show obvious
horizontal banding rather than a seamless floor. The mirror maps the floor *plane* to itself, but
what the face displays is the floor as seen from the *mirrored camera position*, which is a
different viewpoint and generally does not line up. Seamlessness was never predicted; it was
only ever an impression from one axis-aligned frame.

That made the axis-aligned chrome cube suspicious again, so it was measured directly rather than
eyeballed.

**Command:** render, at identical camera/resolution, (a) floor only, (b) matte cube, (c) chrome
cube. Take the cube silhouette from (b); inside it, compare (c) against (a).
**Output/measurement:**
- pixels inside the silhouette **identical to bare floor: 0** (of 123502)
- Pearson **corr(chrome surface, floor behind it) = −0.013**
- mean |chrome − floor| = 47.5, against a floor luminance std of 45.0
- ratio chrome/floor: median 0.889 but IQR 0.333–1.000 — far too wide to be a constant tint

**Hypothesis update:** the chrome faces are structurally *uncorrelated* with whatever lies behind
them. They are not passing the background through, nor showing it scaled by a tint. The pattern
merely shares the floor's two gray levels and rough cell scale, which is what fooled the eye. The
transparency hypothesis is refuted a second time, by an independent measurement.

## Verdict

**H3.2 is not a rendering defect.** Converging evidence:

| Check | Result |
|---|---|
| Occlusion detector (validated on fixtures) | 0% transmission through matte, plastic, metal, gold, chrome |
| Pixels identical to bare floor inside silhouette | 0 of 123502 |
| Correlation of surface with background behind it | −0.013 |
| Flat "hole" faces | background (76,25,51) × chrome tint = (68,22,45) — correct sky reflection |
| Reflection vs. face orientation | changes correctly: axis-aligned → near-seamless, yaw → banded, oblique → tilted checker |
| Dark blob near sphere | cast shadow (uniform ×0.47 on both checker shades) |

The scenes look wrong because a metallic = 1.0 material has almost no diffuse term, and an
infinite self-similar checkerboard under a dark sky is genuinely hard to distinguish from glass
by eye. No shader change is warranted.

**Fixture retained for regression use:** the occlusion detector is the reusable artefact from
this investigation — place a saturated matte sphere behind a subject, assert zero of its
signature colour survives inside the subject's silhouette. It validated cleanly on both a
known-good (opaque cube → 0) and known-bad (unobstructed sphere → 4212) fixture.

## Round 4 — VERDICT WITHDRAWN: per-face shading collapses under a multi-light rig

The "not a defect" verdict above is **retracted**. User inspection of the rotated-chrome renders
flagged two artefacts that survive scrutiny: a **curved** horizon between checker and sky on the
yaw cube, and rectangular reflection blocks on the tilt cube. A planar mirror reflecting a planar
floor must produce a *straight* horizon, so curvature is not explainable as correct optics.

Chasing that led away from the reflection path entirely.

**Command:** DSL cubes, `Material.Matte`, varying one factor at a time; count distinct non-floor
colours in the frame (a flat-shaded cube must show one shade *per visible face*, i.e. three).

| Scene | Lights | Explicit `color` | Rotation | Distinct face shades |
|---|---|---|---|---|
| `LightDown` | 1 × (0,−1,0) | no | none | **3** ✓ (76 / 108 / 255) |
| `LightDownColored` | 1 × (0,−1,0) | yes | none | **3** — byte-identical to above |
| `TwoLightsNegligible` | 2, same direction | no | none | **3** ✓ |
| `TwoLights` | (1,−1,−1)@1.5 + (−1,−0.5,1)@0.5 | no | none | **1** ✗ (76 only) |
| `MatteFlat` | same 2-light rig | yes | none | **1** ✗ |
| `MatteYaw` | same 2-light rig | yes | Y 30° | **1** ✗ |
| `MatteTilt` | same 2-light rig | yes | X 30° Y 20° | **1** ✗ |

**Hypothesis updates:**
- **Rotation is innocent.** `MatteFlat` (zero rotation) collapses identically to the rotated ones.
- **Light *count* is innocent.** Two lights sharing a direction shade correctly.
- **Separate minor defect (still stands):** an explicit DSL `color` is silently ignored when
  `material` is also set — `LightDownColored` is byte-identical to `LightDown`
  (63009/62677/61814 px), which is why the "gold" test cubes rendered grey.

### CORRECTION — the "collapse" was a fault in my test rig, not in the renderer

`Directional.direction` points **toward** the light (`dsl/Light.scala:12`). Verified empirically
by sampling named faces of a single-light cube:

| direction | TOP face | LEFT face | RIGHT face |
|---|---|---|---|
| `(0, +1, 0)` | **(255,255,255)** lit | (76,76,76) | (76,76,76) |
| `(0, −1, 0)` | (76,76,76) | (76,76,76) | (76,76,76) |

So 76 is simply the unlit/ambient baseline, and the doc is correct. The rig I called "collapsing"
used `(1,−1,−1)` and `(−1,−0.5,1)` — **both with negative y, i.e. both lights below the floor**.
Every upward-facing surface was legitimately unlit, so one flat ambient value is the *correct*
result. **There is no multi-light defect.** The CLI never reproduced it because I gave the CLI the
negated (above-floor) vectors, which is the only reason that comparison looked like a DSL/CLI
divergence.

### Does the light rig explain scene 92 anyway?

Scene 92's two lights do both point below the floor, so the question stood on its own. Rendering
the real rig against a negated one:

| rig | gold px | mean luminance on object | distinct shades |
|---|---|---|---|
| as authored (both below floor) | 50899 | 65.2 | 15 |
| negated (above floor) | 33448 | 34.1 | 13 |

The sponge looks materially the same in both; only the *floor* brightness changes, and the
negated version is if anything darker. **Light direction is not the cause of the symptom** —
expected, since `Material.Gold` is metallic ≈ 1.0 and therefore almost pure reflection with a
negligible diffuse term for the lights to act on.

### Standing at end of round 4

Still true: no transmission (round 3), reflections respond correctly to face orientation
(round 3b + octahedron/yaw/tilt). Now also dead: the multi-light hypothesis, the normalization
hypothesis, the rotation hypothesis, and the light-direction hypothesis.

**Still unexplained, and the only hard evidence of a genuine defect:** the *curved* boundary
between checker and sky on the yaw cube's flat vertical face, and the rectangular reflection
blocks on the tilt cube. A planar mirror reflecting a planar floor must yield a straight horizon.
Note that per-face normals are demonstrably flat (a single-light matte cube renders exactly one
colour per face), which makes normal interpolation an unlikely explanation and leaves this open.

## Round 5 — ROOT CAUSE: object rotation does not rotate normals

The user flagged a curved checker/sky boundary on the yaw cube and rectangular reflection blocks
on the tilt cube. Both are real. Zooming to 1600×1200 showed every boundary is made of straight
segments (the *visual* impression of a curve at 700 px was aliasing), but measuring the actual
boundary proved a genuine, smooth curvature.

**Command:** render `ChromeYaw` at 1600×1200; extract the maroon/checker boundary per row; fit.
**Output/measurement:**

| fit | RMS | max \|dev\| |
|---|---|---|
| straight line | 2.31 px | **5.41 px** |
| quadratic | 0.30 px | 0.65 px |

Slope drifts monotonically 0.471 → 0.311; sagitta ≈ 3.8 px. Antialiasing is off, so edges are
hard and 5.4 px is far above noise. A planar mirror reflecting a planar floor must give a
straight boundary, so this is a defect.

**Control — the same measurement on an *unrotated* cube** (low camera so its vertical faces
reflect the horizon; reflected sky is chrome-tinted `(68,22,45)` vs real sky `(76,25,51)`, so the
two separate exactly):

| cube | reflected-horizon straightness |
|---|---|
| unrotated | straight to **0.00 px** RMS over 909 columns |
| yaw 30° | line fit off by 5.41 px |

Rotation about Y leaves faces vertical, so both cases must obey identical reflection geometry.
They do not.

**Confirming experiment — are normals rotated with the geometry?** Matte cube, single overhead
light, tilted about X:

| tilt | shade palette | top-face shade |
|---|---|---|
| 0° | {255, 108, 76, 19} | 255 |
| 30° | {255, 108, 76, 19} | 255 (should be ≈221 = 255·cos30°) |
| 60° | {255, 108, 76, 19} | 255 (should be ≈128 = 255·cos60°) |

Pixel counts change with tilt (77903 → 92823 → 94081), so the geometry really is rotating, but
the shade *values* are byte-identical across all three.

**Root cause: `rotation` transforms vertex positions but leaves vertex normals untouched.** Every
rotated object therefore shades and reflects as though it were unrotated. This explains the
curved/impossible reflection boundaries (the reflection uses a normal that no longer matches the
visible face orientation), sky reflected on faces that cannot see it, and the rectangular blocks —
all without any fault in `reflect()` itself, which rounds 3/3b showed to be correct.

**Scope:** affects every DSL scene using `rotation`, including the two H3.2 scenes after the
fixture edits made this round. Combined with the CLI no-op below, object rotation is broken on
both entry paths: the CLI silently drops it, and the DSL applies it to geometry only.

## Round 6 — fix applied, and a correction to round 5's evidence

**Fix (`optix-jni`, `shaders/hit_triangle.cu`):** both `getTriangleGeometry` overloads now carry
the interpolated vertex normal to world space:

```cuda
normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));
```

Vertex normals are stored in object space; OptiX applies the instance transform
(`addTriangleMeshInstance(transform, …)`, a real 12-float instance transform — `OptiXWrapper.cpp:2266`)
to the geometry during traversal but never to the normals. Before this, `hit_curve.cu` was the
**only** shader in the repo calling that intrinsic.

**Correction — round 5's headline measurement was invalid.** The tilt test used a directional
light of intensity **2.0**, which saturates: `cos(60°) × 2.0 = 1.0`, still clamping to white. So a
top face was pinned at 255 for every tilt from 0° to 60° *regardless of how normals were handled*.
The identical palettes proved nothing. Re-run at intensity 1.0 the palettes do vary with tilt. The
conclusion "geometry rotates but normals do not" was reached from a test that could not have
detected the difference — the fix stands on the code reading, not on that measurement.

**Plumbing verified before trusting any result:** forcing `normal` to a constant in the
SBT-data variant changed nothing (byte-identical pixel counts), but forcing it in the
explicit-pointer/IAS variant did change the render. So the IAS overload is the live path for these
scenes, and the fix sits on executed code. (Also confirmed the runtime extracts the PTX from the
jar to `target/native/x86_64-linux/bin/optix_shaders.ptx` relative to the *working directory* —
`OptiXRenderer.scala:717-723` — so stale copies elsewhere in the tree are not used.)

**Acceptance test — the artefact the user reported is gone.** Chrome cube, yaw 30°, 1600×1200,
measuring where reflected sky (maroon) appears:

| | maroon y-range | vertical-face sky wedges |
|---|---|---|
| before fix | 210 … **1110** | present (impossible: a downward-looking camera cannot see sky reflected in a vertical mirror) |
| after fix | 210 … **427** | **gone** — only the upward-facing top face reflects sky |

The region containing the curved boundary (y 620…1000, straight-line fit off by 5.41 px) no longer
exists at all.

**Still to do:** re-measure a reflected horizon end-to-end on a rotated object for a clean
straightness number, and check the other analytic shaders (`hit_sphere.cu`, `hit_cone.cu`,
`hit_cylinder.cu`) which compute normals from object-space data and never call the transform
either — they are likely to carry the same defect for rotated instances.

## DEFERRED DEFECT — CLI object rotation is a silent no-op

**Found incidentally while building fixtures for this investigation. Not part of H3.2; must be
raised when the H3.2 fixtures are cleaned up and committed.**

`--rot-x` / `--rot-y` / `--rot-z` have no effect on geometry supplied via `--objects`.
`y_chrome_roty30.png` (`--rot-y 30`) and `y_chrome_rotxy.png` (`--rot-y 30 --rot-x 20`) are
**pixel-identical** to the unrotated cube — verified with an exact array comparison, not by eye.
The flags are accepted silently; there is no warning and no error. Object rotation is reachable
only through the DSL `rotation` parameter.

This is significant on its own: every CLI scene and every `manual-test.sh` / `integration-tests.sh`
entry that passes `--rot-*` alongside `--objects` is silently rendering an unrotated object, so
those tests assert against references that never exercised rotation. Needs its own investigation
and fix, plus an audit of which scripted scenes pass `--rot-*`.

**Why this matters for H3.2:** scene 84 (`MengerShowcase`) uses a **three**-point directional rig
and scene 92 (`PulsingSponge`) uses **two** — both in the affected class. An object whose faces
all shade to one flat dark value loses the per-face contrast that reads as solid form, which is a
far better explanation of "the metal looks transparent" than anything found in the reflection
path. The earlier measurements in rounds 3/3b remain valid on their own terms (there really is no
transmission), but they were answering the wrong question.

**Open, not yet explained:** the curved horizon and rectangular reflection blocks. These may be
downstream of the same shading defect or independent; not yet established.

**Note on invalidated reasoning:** round 3's inference "matte cube renders flat ⇒ normals are
flat per face" was unsound. Under the multi-light rig the diffuse result is orientation-*independent*,
so it carries no information about the normals at all. Any future normal-related test must use a
single-light rig.
