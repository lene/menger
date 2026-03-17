# Sprint 12: t-Parameter Animation System

**Sprint:** 12 - t-Parameter Animation
**Status:** Complete
**Branch:** `feature/sprint-12`
**Dependencies:** Sprint 10 (Scala DSL), Sprint 11 (LibGDX wrapper, thin-film)

---

## Goal

Animate DSL-specified scenes using a free parameter `t`. Animated scenes define `def scene(t: Float): Scene` instead of `val scene: Scene`. The renderer evaluates `scene(t)` for each frame, doing a full scene rebuild.

## Success Criteria

- [x] `LoadedScene` ADT distinguishing `Static` vs `Animated` scenes
- [x] `SceneLoader` auto-detects animated scenes via reflection
- [x] CLI options: `--t`, `--start-t`, `--end-t`, `--frames`
- [x] CLI validation: mutual exclusivity, `--scene` and `--optix` requirements
- [x] `AnimatedOptiXEngine` with per-frame scene rebuild
- [x] `TAnimationConfig` with linear t interpolation
- [x] `SceneConverter` utility for reusable DSL-to-config conversion
- [x] Example animated scenes: `OrbitingSphere`, `PulsingSponge`
- [x] Backward compatibility: static `val scene` still works
- [x] All tests pass (27 new tests, 1599+ total)
- [x] Integration tests for freeze-frame and multi-frame animation
- [x] Documentation updated (USER_GUIDE.md, CHANGELOG.md)

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Animation model | `t`-parameter (functional) | More expressive than keyframes; leverages compiled Scala |
| `t` semantics | User-defined range (`--start-t`, `--end-t`) | Maximum flexibility |
| Scene function signature | `def scene(t: Float): Scene` | Minimal, clean |
| Per-frame approach | Full scene rebuild | Simplest; ray tracing dominates frame time anyway |
| Backward compatibility | Yes, static `val scene` still works | Low implementation cost via reflection fallback |

---

## CLI Usage

```bash
# Freeze-frame at t=0.5
menger --optix --scene examples.dsl.OrbitingSphere --t 0.5 --save-name orbit.png --headless

# 100-frame animation, t goes from 0 to 6.28
menger --optix --scene examples.dsl.OrbitingSphere --start-t 0 --end-t 6.28 --frames 100 --save-name orbit_%04d.png --headless
```

## DSL Usage

```scala
object OrbitingSphere:
  def scene(t: Float): Scene =
    val x = 2f * math.cos(t).toFloat
    val z = 2f * math.sin(t).toFloat
    Scene(
      camera = Camera(position = (0f, 3f, 6f), lookAt = (0f, 0f, 0f)),
      objects = List(Sphere(pos = (x, 0f, z), material = Material.Chrome, size = 0.5f)),
      lights = List(Directional(direction = (1f, -1f, -1f), intensity = 2.0f)),
      plane = Some(Plane(Y at -1.5, color = "#FFFFFF"))
    )
```

---

## Tasks

### Task 12.1: LoadedScene ADT + SceneLoader Extension

**Status:** Complete

**Goal:** SceneLoader detects `def scene(t: Float)` vs `val scene` via reflection.

**Created:**
- `menger-app/src/main/scala/menger/dsl/LoadedScene.scala` -- sealed trait with `Static` and `Animated` cases

**Modified:**
- `menger-app/src/main/scala/menger/dsl/SceneLoader.scala` -- returns `Either[String, LoadedScene]`, reflection detects animated vs static

**Tests:**
- Extended `SceneLoaderSuite` with static and animated loading tests
- Created `TestAnimatedSceneObject` test fixture

---

### Task 12.2: CLI Options for t-Parameter Animation

**Status:** Complete

**Goal:** Add `--t`, `--start-t`, `--end-t`, `--frames` to CLI.

**Modified:**
- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` -- added 4 new options in "Scene Animation (t-parameter)" group
- `menger-app/src/main/scala/menger/cli/CliValidation.scala` -- added `registerTAnimationValidations()` with mutual exclusivity rules

**Tests:**
- New `TAnimationCLIOptionsSuite` covering all validation rules

---

### Task 12.3: AnimatedOptiXEngine + TAnimationConfig

**Status:** Complete

**Goal:** Frame loop that re-evaluates `scene(t)` per frame, rebuilds OptiX scene, renders, saves.

**Created:**
- `menger-app/src/main/scala/menger/engines/TAnimationConfig.scala` -- config with `tForFrame()` interpolation
- `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` -- frame loop engine
- `menger-app/src/main/scala/menger/dsl/SceneConverter.scala` -- reusable DSL-to-config conversion

**Tests:**
- `TAnimationConfigSuite` covering interpolation, edge cases (1 frame, equal start/end)

---

### Task 12.4: Main.scala Wiring

**Status:** Complete

**Goal:** Wire animated scene detection and engine dispatch into Main.

**Modified:**
- `menger-app/src/main/scala/Main.scala` -- pattern match on `LoadedScene.Static` vs `LoadedScene.Animated`, dispatch to `AnimatedOptiXEngine` or freeze-frame

---

### Task 12.5: Example Animated Scenes

**Status:** Complete

**Goal:** Create example scenes demonstrating the `def scene(t: Float)` pattern.

**Created:**
- `menger-app/src/main/scala/examples/dsl/OrbitingSphere.scala` -- sphere orbiting origin
- `menger-app/src/main/scala/examples/dsl/PulsingSponge.scala` -- sponge level varies with `t`

**Modified:**
- `menger-app/src/main/scala/examples/dsl/SceneIndex.scala` -- references new objects

**Tests:**
- Extended `ExampleScenesSuite` with animated scene tests

---

### Task 12.6: Shell Tests + Documentation

**Status:** Complete

**Goal:** Integration tests and documentation updates.

**Modified:**
- `scripts/integration-tests.sh` -- added `test_t_animation()` with freeze-frame and multi-frame tests
- `scripts/manual-test.sh` -- added animated scene entries for static and interactive tests
- `CHANGELOG.md` -- added t-parameter animation entries under `[Unreleased]`
- `docs/USER_GUIDE.md` -- added "Animated Scenes (t-Parameter)" section

---

### Task 12.7: Update Sprint Plans and Roadmap

**Status:** Complete

**Goal:** Update sprint planning documents to reflect the new animation approach.

**Modified:**
- `docs/sprints/SPRINT12.md` -- replaced with t-parameter animation plan (this document)
- `docs/sprints/SPRINT13.md` -- replaced with old Sprint 12 content (Visual Quality & Materials)
- `docs/sprints/SPRINT14.md` -- replaced with trimmed plan (Video Output & Visual Enhancements)
- `ROADMAP.md` -- updated planned sprints section

---

## Implementation Order

```
12.1 (LoadedScene + SceneLoader)  --+
                                    +--> 12.3 (AnimatedOptiXEngine) --> 12.4 (Main wiring) --+
12.2 (CLI options)  ----------------+                                                        +--> 12.6 (docs) + 12.7 (sprint plans)
                                                                                             |
12.5 (Example scenes) --- depends on 12.1 --------------------------------------------------+
```

Tasks 12.1 and 12.2 were developed in parallel. Tasks 12.6 and 12.7 were done in parallel at the end.

---

## Key Files

### New Files Created

| File | Purpose |
|------|---------|
| `menger-app/src/main/scala/menger/dsl/LoadedScene.scala` | ADT: Static vs Animated scene |
| `menger-app/src/main/scala/menger/engines/TAnimationConfig.scala` | Animation config (startT, endT, frames) |
| `menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala` | Frame loop engine |
| `menger-app/src/main/scala/menger/dsl/SceneConverter.scala` | Scene-to-config extraction |
| `menger-app/src/main/scala/examples/dsl/OrbitingSphere.scala` | Example: orbiting sphere |
| `menger-app/src/main/scala/examples/dsl/PulsingSponge.scala` | Example: pulsing sponge |

### Modified Files

| File | Change |
|------|--------|
| `menger-app/src/main/scala/menger/dsl/SceneLoader.scala` | Return `LoadedScene`, detect animated scenes via reflection |
| `menger-app/src/main/scala/menger/MengerCLIOptions.scala` | Add `--t`, `--start-t`, `--end-t`, `--frames` |
| `menger-app/src/main/scala/menger/cli/CliValidation.scala` | Add t-animation validation rules |
| `menger-app/src/main/scala/Main.scala` | Dispatch animated scenes to AnimatedOptiXEngine |
| `menger-app/src/main/scala/examples/dsl/SceneIndex.scala` | Register new example scenes |

---

## Verification

```bash
# 1. Compile
sbt compile

# 2. Run all tests
sbt test

# 3. Code quality
sbt "scalafix --check"

# 4. Freeze-frame test
sbt "mengerApp/run --optix --scene examples.dsl.OrbitingSphere --t 0.5 --save-name /tmp/orbit-test.png --headless"

# 5. Multi-frame test (3 frames for speed)
sbt "mengerApp/run --optix --scene examples.dsl.OrbitingSphere --frames 3 --start-t 0 --end-t 1 --save-name /tmp/orbit_%04d.png --headless"

# 6. Static scene backward compat
sbt "mengerApp/run --optix --scene glass-sphere --save-name /tmp/glass-test.png --headless"

# 7. Integration tests
bash scripts/integration-tests.sh
```
