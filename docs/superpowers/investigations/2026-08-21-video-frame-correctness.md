# Investigation: is the video frame correct? (manual-test #15)

Two scenes: DSL VideoTextureCube (96), DSL EnvMapVideoSponge (97). Original report: a
verification task, not a confirmed defect.

## Stage 0 — Context — 2026-08-21

**VideoTextureCube.** Its docstring already documents the intended behaviour: "rectangular
video texture decoded as a still initial frame." Extracted the fixture's two frames with
`ffmpeg` for ground truth: frame 1 = R/G/B/W quadrants (row-major), frame 2 =
cyan/magenta/yellow/black. Rendered the scene headless and visually confirmed the R/G/B/W
palette is present (bilinear-blurred across the 2x2 texture, camera-angle and lighting
distort exact corner mapping but the palette and rough quadrant layout match frame 1, not
frame 2 or garbage).

**Command:**
```
./menger-app/target/universal/stage/bin/menger-app --headless --scene examples.dsl.VideoTextureCube \
  --texture-dir menger-geometry/src/test/resources/ -s /tmp/video-texture-cube.png
```
**Output/measurement:** frame 1's palette present (visual inspection, cropped/enlarged render).
**Hypothesis update:** VideoTextureCube is correct as documented. No bug, no fix needed.

## Stage 0 — Context, EnvMapVideoSponge — 2026-08-21

`two-frame-equirect-rgba.mov` fixture: frame 1 = solid red `(253,0,0)`, frame 2 = solid blue
`(0,0,254)` — trivially distinguishable by a single corner pixel sample.

**Command (three single-still previews):**
```
./menger-app/target/universal/stage/bin/menger-app --headless --scene examples.dsl.EnvMapVideoSponge \
  --texture-dir menger-geometry/src/test/resources/ --t <0|0.5|0.99> -s /tmp/envmap-t<t>.png
```
**Output/measurement:** background corner pixel = `(127,0,0)` (Reinhard-tonemapped red) at
**all three** `t` values, including `t=0.99`.
**Hypothesis update:** the background is stuck on frame 1 regardless of `--t`. Real bug, not
a clean verification.

**Command (true animated sweep, for comparison):**
```
./menger-app/target/universal/stage/bin/menger-app --headless --scene examples.dsl.EnvMapVideoSponge \
  --texture-dir menger-geometry/src/test/resources/ --frames 8 --start-t 0 --end-t 1 \
  --save-name /tmp/envanim/env_video_%04d.png
```
**Output/measurement:** frames 0-2 red, frames 4-6 blue (correctly alternates as `t` sweeps
0→1, matching the fixture's 1-second/2-frame duration at 2fps with looping).
**Hypothesis update:** the true `--frames` sweep path works correctly. The bug is isolated to
the single-still `--t` preview path — different code path, same DSL scene.

## Root cause — 2026-08-21

`--log-level DEBUG` on the single-still run showed exactly one texture upload, keyed
`env-video:...|time=TProgress|repeat=Loop|offset=0.0|fps=2.0` — the cache key has no
resolved-time component at all. Traced the call chain: `InteractiveEngine.create()`
(`InteractiveEngine.scala:304-306`) calls `TextureManager.loadInitialEnvMapVideo`, whose
`loadInitialEnvMapVideoData` (`TextureManager.scala:239-258`, prior to fix) called
`loader.frameAt(envMapVideo.playback.startOffset)` directly — completely bypassing
`VideoPlaybackTime.sampleTime`, the time-mapping logic that already existed and worked
correctly, but only in `WithAnimation.VideoFrameSource.frameAt`
(`WithAnimation.scala:515-523`), used exclusively by the true multi-frame `--frames` sweep
engines. `InteractiveEngine` has no `renderT` concept at all — `--t` is consumed once in
`Main.scala`'s `createSceneBasedEngine` to call `fn(freezeT)` and build the DSL `Scene`, then
discarded; nothing downstream knew what `t` had been.

`EnvMapVideoSponge`'s own docstring advertises `--t 0.5` as a valid preview usage
(alongside the `--frames` sweep), so this is a real defect, not an intentional
"previews always show the start frame" design (unlike `VideoTextureCube`, which documents
exactly that as its intended behaviour).

## Checkpoint — user confirmed: fix it now

## Stage 3 — Fix — 2026-08-21

- `TextureManager.loadInitialEnvMapVideoData` / `loadInitialEnvMapVideo`: added a
  `renderT: Float = 0f` parameter; resolve the frame via
  `VideoPlaybackTime.sampleTime(playback, renderT, AnimationTimeRange(0f, 1f), loader.durationSeconds)`
  instead of `loader.frameAt(startOffset)`. `AnimationTimeRange(0f, 1f)` is a placeholder —
  only the `AnimationRange` time mapping consults it, and a single-still preview (no
  `--start-t`/`--end-t`) has no real sweep range to give it; `TProgress`/`TSeconds` ignore it
  entirely.
- `InteractiveEngine`: added a `renderT: Float = 0f` constructor parameter, threaded into the
  `TextureManager.loadInitialEnvMapVideo` call. Default keeps the `--objects` CLI path
  (which never sets `envMapVideo`) unaffected.
- `Main.scala`: `createSceneBasedEngine`'s `Right(loadedScene)` branch now captures `freezeT`
  once and passes it through `createOptiXEngineFromDslScene(opts, dslScene, freezeT)` →
  `InteractiveEngine(engineConfig, opts.userSetMaxInstances, renderT = freezeT)`.

**Verification:**
```
--t 0     -> corner (127,0,0)  [red,  was already correct]
--t 0.99  -> corner (0,0,127)  [blue, was (127,0,0) before the fix]
```
Re-ran the `--frames 8 --start-t 0 --end-t 1` sweep after the fix: unchanged (red/red/blue/
blue), confirming no regression to the already-working animated path.

**Regression test:** `TextureManagerSuite` — new test asserts `loadInitialEnvMapVideoData`
with `renderT=0` and `renderT=0.99` decode different frame bytes from the same `EnvMapVideo`
(would have failed before the fix — both always returned frame 1).

**Reference-image impact:** `integration-tests.sh`'s existing "DSL EnvMapVideoSponge" scene
uses `--t 0.5`, which sits exactly on the frame-1/frame-2 boundary of the 1-second/2fps
fixture — before the fix it always resolved to frame 1 (red); after the fix, `t=0.5` resolves
to frame 2 (blue), matching `VideoLoaderSuite`'s own documented boundary behaviour
(`frameAt(0.5) shouldBe SecondFrame`). Regenerated `DSL_EnvMapVideoSponge.png`; full
sequential suite re-run confirms this is the only reference affected.

## Minimum artefacts checklist

- [x] Regression test exists (`TextureManagerSuite`, permanent)
- [x] Fix commit passes that test
- [x] This note records per-step commands, measurements, root-cause narrative
- [x] Prior description ("needs investigation") corrected in `ManualTestNeedFixing.md`
- [x] No `CODE_IMPROVEMENTS.md` entry existed for this
- [x] No residual defects found (VideoTextureCube confirmed correct; `--frames` sweep
      confirmed unaffected)
