# Ground/sky flip when camera height changes — investigation

Sprint 36 Phase H3.1. Prior report (`ManualTestNeedFixing.md`): "ground and sky flip when
changing the height from which a scene is viewed" — scenes 86, 88-95, 98, 100, 102, 103, 165,
"probably many more". Not yet diagnosed at plan time (Sprint 36 Phase H plan, H3.1).

## Stage 0 — Context

**Command:**
```
BIN=./menger-app/target/universal/stage/bin/menger-app
xvfb-run -a "$BIN" -o --headless -s <out>.png \
  --objects type=sphere:material=chrome --plane y:-2 \
  --camera-pos 0,<Y>,5 --camera-lookat 0,-1,0
```
Swept `<Y>` = 1.5, 0, -1.5, -2.5 (plane at y=-2; camera passes from well above, to just
above, to just below the plane).

**Output/measurement:** Four renders, `/tmp/.../scratchpad/flip/y_{1.5,0,-1.5,-2.5}.png`.

- `y=1.5`: lookAt -1 makes camera look down enough that no sky is visible in frame at all
  (whole frame is checker floor + sphere). Not a clean horizon shot — framing artifact of
  this particular lookAt choice, not the bug.
- `y=0` (2 units above plane): clean horizon — magenta sky fills top ~1/6, checker floor
  fills the rest below a sharp horizon line. Sphere's upper hemisphere reflects the sky
  (flat magenta), lower hemisphere reflects the checker floor. This looks correct/expected.
- `y=-1.5` (0.5 units above plane, close to it): horizon still normal — sky top, floor
  bottom, same as y=0. But the **sphere's reflection has changed**: the entire visible
  sphere now shows the checker-floor reflection; no part of it reflects the flat sky color
  any more.
- `y=-2.5` (0.5 units below plane): the checker floor pattern now fills the frame on
  **both** sides of the horizon line — the region that previously showed the flat sky
  color is now also checkered, with only a small disc of flat colour left directly behind
  the sphere.

**Hypothesis update:** Two distinct visual changes as camera height decreases through the
plane's y-level, not obviously the same mechanism:
1. Sphere reflection hemisphere split disappears/inverts near the plane height (y=-1.5).
2. Sky region gets replaced by floor-checker fill once camera crosses below plane height
   (y=-2.5).

Neither has an assigned cause yet. Renders sent to user for description sign-off
(Checkpoint 1) before proceeding to Stage 1 (detector).

**Correction:** user rejected this repro — "no. i am giving you 4 images of scene 165."
Synthetic sphere+plane sweep above did not match what was actually reported. Discarded;
real repro below uses the user's own screenshots of scene 165 (`TrefoilKnot`,
`manual-test.sh:650`, array index 165).

## Stage 0b — Real repro (TrefoilKnot, scene 165)

User supplied 4 screenshots from their own interactive session: (1) default camera position,
(2) tilted toward the flip direction, (3) tilted further, just before the flip, (4) tilted a
tiny bit more, just after the flip. Confirmed via follow-up questions:
- Tilt was a left-drag (orbit), not a different input mode.
- Image 1: magenta = sky (background), black = floor.
- Image 4 (just after flip): knot still visible, distinguishable, on a black background —
  not a fully solid frame.

`--scene` (DSL) mode does not honor `--camera-pos`/`--camera-lookat`/`--camera-up` — camera
comes entirely from the DSL scene's own `camera` field (`Main.scala` `createSceneBasedEngine`
→ `SceneConverter.convert`). TrefoilKnot's exact camera (eye `(0,0,8)`, lookAt origin,
`planes = List(Plane(Y at -3.0, color = "#CCCCCC"))`) could not be swept directly via CLI —
reproduced the underlying mechanism on a `--objects`-based scene instead, matched against
real telemetry (below).

**Diagnostic instrumentation** (temporary, `ponytail:`-marked, added and later removed):
`OrbitCamera.debugElevation`/`debugAzimuth` accessors, bumped `OptiXCameraHandler`'s camera
update log to INFO including elevation/azimuth. User reproduced the bug live and pasted the
log:

```
elevation=-20.399984 eye=(3.157,-2.789,6.801) ... through ... elevation=-24.3 eye.y≈-3.29
```

`eye.y` crosses the plane's `y=-3` between elevation -21.9° and -22.8° — exactly where the
user saw the flip.

**Two hypotheses tested and ruled out** by direct render sweep (elevation 0→89°, both plane
normal directions): elevation approaching the ±89° pole causing a degenerate/gimbal-lock
camera basis. Zero discontinuities found in either sweep — not the mechanism.

**Confirmed mechanism:** independent headless reproduction on a plain sphere+plane scene,
sweeping elevation across the mathematically-predicted crossing point (elevation ≈ -22.024°
for distance=8, plane value=-3):
- elevation=-20° (eye.y=-2.74, above plane): frame ~100% background color (magenta).
- elevation=-22.5° (eye.y=-3.06, just past crossing): frame ~100% plane color (gray).

Instant full-frame flip, not a gradually-rising horizon. Renders sent to user
(`cross_-20.png`, `cross_-22.5.png`); user confirmed match — **Checkpoint 1 passed.**

**Why the knot stays visible through the flip (user's follow-up question):** the knot is
real BVH-hit-tested geometry; the plane at this point is legacy miss-shader-only
(`miss_plane.cu`) — it only ever supplies a fallback color for rays that hit *no* real
geometry, so it can never occlude anything, regardless of camera position. Confirmed by
reading `PlaneConfigurer.scala` (DSL and CLI planes both route through the same legacy
`addPlane*` calls) and the file header of `miss_plane.cu` itself ("first-class geometry
plane is `hit_plane.cu`, legacy miss path kept for `--plane` back-compat"). This explains
both halves of the symptom in one mechanism: background flips (miss-shader-only, sees the
plane) / knot stays visible (BVH-hit, plane-blind) — not two separate bugs.

## Stage 0c — Second finding: plane never occludes real geometry

User's follow-up ("if the camera is below the plane, why is the knot visible at all") led to
a second, broader defect, same root cause, bigger blast radius: **the legacy plane can never
occlude real geometry, at any camera angle** — not just during the elevation flip. Verified
directly: rendered a chrome sphere at `pos=0,-6,0` (4 units on the far side of a
`--plane y:-2`, i.e. behind it from the camera), camera above looking down through the
plane's y-level toward the sphere.

```
--objects type=sphere:pos=0,-6,0:size=1.5:material=chrome --plane y:-2 \
--camera-pos 0,0,10 --camera-lookat 0,-6,0 --camera-up 0,1,0
```

`behind_plane.png`: sphere renders fully, unoccluded, floating in front of the checkerboard
— should have been fully hidden if the plane were real geometry. User confirmed this reading
("that opens an entirely new problem... a transparent ground plane is not realistic in any
way") and asked to fold this into the H3.1 writeup and fix now rather than opening a
separate investigation.

## Stage 1/2 — Root cause and fix

Root cause: `PlaneConfigurer.configurePlanes` (the production path for both `--plane` CLI and
DSL `planes`) used the legacy miss-shader plane API (`addPlaneNative` /
`addPlaneSolidColor(WithMaterial)Native` / `addPlaneCheckerColors(WithMaterial)Native`) —
never enters the BVH, never depth-tested, only a fallback color for missed rays.

A first-class, real-geometry plane path already existed and was fully implemented but unused
in production: `hit_plane.cu` (intersection + closest-hit + shadow programs, "since Sprint
19" per its own header), wired end-to-end through `addPlaneInstanceNative` →
`OptiXRenderer.addPlaneInstance` (`OptiXPlaneApi.scala`) → already used by
`PlaneSceneBuilder` for `type=plane:...` object specs (explains why scenes 126/127 in the
bug list, which use that path, never showed the flip). Fix: switch `PlaneConfigurer` to call
`addPlaneInstance` instead of the legacy API.

**Ordering hazard, found while implementing:** planes now share the same IAS instance space
as scene objects, so every `renderer.clearAllInstances()` call site that previously ignored
planes (because they lived in a separate legacy array) now needs to re-add them. Three call
sites fixed:
- `InteractiveEngine.rebuildScene()` (4D-rotation live rebuild) — planes would vanish on the
  first rotation drag.
- `CliAnimationEngine.render()` — planes would vanish from frame 2 of `--animate` onward.
- `WithAnimation.render()` — `configurePlanes` was unconditional even when the 4D/video fast
  path skips `clearAllInstances()`, which would have duplicated plane instances every frame.
  Moved inside the `if !fastPathTaken` block so it only re-adds planes exactly when the
  instance space was actually cleared.

## Stage 3 — Verification

Re-ran the occlusion test post-fix: `behind_plane_fixed.png` — sphere behind the plane is now
correctly hidden. Sanity-checked a standard above-floor scene (`normal_sphere.png`) — no
regression, chrome sphere on checkerboard floor renders as before.

Re-ran the elevation-crossing pair with the fix: `fixed_elev_-20.png` (pre-crossing, sphere
visible normally against magenta) vs `fixed_elev_-22.5.png` (post-crossing, **100% uniform
gray — sphere now occluded too**, triggered the render-health uniform-output guard,
re-rendered with `--allow-uniform-render` to confirm). This is the correct behavior for an
infinite opaque plane viewed from underneath — realistic, but changes TrefoilKnot's specific
symptom: the knot will now also disappear behind the floor once the eye crosses below it,
rather than staying visible while only the background recolors.

**User confirmed this is the desired outcome** ("solid gray is the best solution here") — no
camera/floor collision guard needed as a follow-up. Investigation closed:
- Background-flip mechanism (Stage 0b): root-caused, not itself a bug — inherent to crossing
  an infinite plane's y-level; instant transition is unavoidable without a collision guard,
  which the user declined.
- Transparency defect (Stage 0c): fixed — `PlaneConfigurer` now uses real occluding geometry.

**Follow-up, not actioned this session:** temporary `ponytail:` diagnostic instrumentation
(`OrbitCamera.debugElevation`/`debugAzimuth`, `OptiXCameraHandler`'s INFO-level camera log)
still needs removing. User's separate suggestion — give TrefoilKnot's floor a checker
pattern instead of solid gray, for orientation — not yet actioned.
