# Sprint 22: HDR Environment Maps

**Sprint:** 22 - HDR Environment Maps
**Status:** Not Started
**Estimate:** ~9 hours
**Branch:** `feature/sprint-22`
**Dependencies:** Sprint 20 (texture pipeline, HDR float upload), Sprint 21 (4D fractals)

---

## Goal

Make the existing equirectangular environment map GPU infrastructure usable from the DSL.
A user should be able to specify a `.hdr` file as the scene background and render fractals
(including 4D fractals from Sprint 21) animated in front of it via `scene(t)`.

The env map is background only in this sprint — it does not illuminate objects.
Image-based lighting (IBL) is Sprint 23.

## Success Criteria

- [ ] `Scene(..., envMap = Some("path/to/panorama.hdr"))` renders the HDR as background
- [ ] HDR values > 1.0 display correctly with tone mapping (no clipped white patches)
- [ ] Tone mapping operator and exposure configurable in DSL
- [ ] 4D Menger sponge and Sierpinski analogs (Sprint 21) animate in front of HDR background
- [ ] Example scene files demonstrate 3D rotation + 4D rotation + fractional level sweep with HDR background
- [ ] All tests pass

---

## Tasks

### Task 22.1: Wire DSL Environment Map

**Estimate:** 2h

The GPU side is fully implemented (Sprint 20): `miss_plane.cu:sampleEnvMap()` samples an
equirectangular texture, `Params.env_map_texture` holds the CUDA texture object,
`setEnvironmentMap(textureIndex)` activates it. Only the Scala/DSL wiring is missing.

**Implementation:**
1. Add `envMap: Option[String] = None` to `Scene` case class
   (`menger-app/src/main/scala/menger/dsl/Scene.scala:26`)
2. Wire through `EnvironmentConfig` (`menger-app/src/main/scala/menger/config/EnvironmentConfig.scala:12`
   — `envMap` field already defined there)
3. In `SceneConfigurator` (or equivalent scene setup path): after scene load, if `envMap` is
   set, call `renderer.uploadTextureFromFile(path)` then `renderer.setEnvironmentMap(index)`
4. Smoke test: render with an HDR env map, assert miss-shader pixels differ from solid background

---

### Task 22.2: Tone Mapping

**Estimate:** 3h

HDR environment maps contain values > 1.0 for bright light sources (sun, fires, lamps).
Without tone mapping, these display as clipped white. A tone mapping pass converts HDR
linear values to display range [0,1] before writing to output buffer.

**Implementation:**
1. Add tone mapping DSL type:
   ```scala
   sealed trait ToneMapping
   object ToneMapping:
     case class Reinhard(exposure: Float = 1.0f) extends ToneMapping
     case class ACES(exposure: Float = 1.0f)     extends ToneMapping
     case object None                             extends ToneMapping
   ```
   Add `toneMapping: ToneMapping = ToneMapping.Reinhard()` to `Scene` case class.

2. Pass `exposure`, `tonemap_operator` to `Params` struct in `OptiXData.h`.

3. Apply in raygen shader after accumulation, before `optixGetFrameBuffer()` write:
   - Reinhard: `c = c * exposure / (1 + c * exposure)`
   - ACES approximation: standard filmic curve

4. `exposure` bindable as `scene(t)` parameter for animated fade-in/fade-out effects.

**Default:** `ToneMapping.Reinhard(exposure = 1.0f)` — preserves existing look for
non-HDR scenes (values already in [0,1] are minimally affected).

---

### Task 22.3: Fractal Animation Example Scenes

**Estimate:** 2h
**Depends on:** 22.1, Sprint 21 (21.1–21.4)

Verify that 4D Menger sponge and Sierpinski analogs from Sprint 21 work with
`Projection4DSpec` via `scene(t)`, and write demo scenes combining all animated parameters.

**Scenes to create:**
1. `examples/dsl/FractalWithHDR.scala`
   - HDR background (`studio.hdr` or similar creative-commons panorama)
   - 4D Menger sponge sweeping `fractionalLevel` 1.0 → 4.0 via t
   - Simultaneous 4D rotation (rotXW, rotYW swept over t)
   - Tone mapping enabled

2. `examples/dsl/SierpinskiHDRRotation.scala`
   - HDR background
   - 4D Sierpinski analog with combined 3D rotation (Y-axis) + 4D rotation (XW plane)
   - Fractional level held constant at 2.5

These also serve as integration tests confirming Sprint 21 animation infrastructure works
end-to-end with the new env map feature.

---

### Task 22.4: Documentation

**Estimate:** 2h

- User guide section: "HDR Environment Maps"
  - How to specify an env map in DSL
  - Supported formats: `.hdr` (32-bit float equirectangular)
  - Tone mapping options and exposure control
  - Note: env map is background only; for object illumination see Sprint 23 (IBL)
- Sprint retrospective
- CHANGELOG.md update

---

## Summary

| Task | Description | Estimate | Dependencies |
|------|-------------|----------|--------------|
| 22.1 | Wire DSL environment map | 2h | Sprint 20 texture pipeline |
| 22.2 | Tone mapping (Reinhard + ACES, exposure) | 3h | — |
| 22.3 | Fractal animation example scenes | 2h | 22.1, Sprint 21 |
| 22.4 | Documentation | 2h | All |
| **Total** | | **~9h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] Example renders committed showing fractals against HDR backgrounds

---

## Notes

### What Is Not In This Sprint

- **IBL (image-based lighting):** env map illuminates background only; objects still lit by
  DSL lights. Full IBL — importance-sampled environment illumination — is Sprint 23.
- **Video/movie backgrounds:** animated `.mp4` backgrounds via ffmpeg are Sprint 24.
- **Light extraction:** no automatic extraction of light sources from HDR. Users place
  standard DSL lights (`Point`, `AreaLight`, `Directional`) manually to match HDR light sources.

### HDR Format

Only `.hdr` (Radiance RGBE) is supported. The `uploadTextureFloat()` function
(`OptiXWrapper.cpp:2477–2532`) already handles this via `stbi_loadf()`. EXR is not supported
(would require OpenEXR SDK).

### Existing GPU Infrastructure (No Changes Needed)

- `miss_plane.cu:sampleEnvMap()` — equirectangular UV mapping from ray direction
- `Params.env_map_enabled`, `Params.env_map_texture` — already in `OptiXData.h`
- `setEnvironmentMapNative()` JNI binding — already in `OptiXWrapper.cpp`
- `uploadTextureFloat()` — 32-bit HDR texture upload, full float range
