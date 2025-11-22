# TODO

## Roadmap

**Full details:** [optix-jni/ENHANCEMENT_PLAN.md](optix-jni/ENHANCEMENT_PLAN.md)

**Detailed plans:**
- [Sprint 5 Plan](optix-jni/SPRINT_5_PLAN.md) - Triangle Mesh + Cube + Polish
- [Sprint 6 Plan](optix-jni/SPRINT_6_PLAN.md) - Full Geometry Support (IAS + Sponge)
- [Sprint 7 Plan](optix-jni/SPRINT_7_PLAN.md) - Materials (UV, textures, presets)

### Sprint 5: Triangle Mesh Foundation + Cube

**Core:**
- Triangle mesh infrastructure (`OptixBuildInputTriangleArray`)
- JNI interface for vertex/index buffers
- Triangle closest-hit shader with refraction
- `--object cube` CLI integration

**Polish (Step 5.5):**
- Fix PTX packaging for distribution
- Encapsulate render options (CLI grouping + internal config objects)
- Disable runtime resolution change on window resize
- Print CLI help on errors

### Sprint 6: Full Geometry Support

- Instance Acceleration Structure (IAS) for multi-object scenes
- Per-object transforms (position, rotation, scale)
- Sponge mesh export from `SpongeBySurface`
- Multiple `--object` flags

### Sprint 7: Materials

- UV coordinates in vertex format
- Texture upload and sampling
- Material presets (glass, metal, matte)
- CLI: `--material` flag
- **🎯 v0.5 Milestone: Full 3D Support**

### Sprint 8-11: Future

- **Sprint 8:** Sponge Mesh Export refinement
- **Sprint 9-10:** Animation (keyframes, transforms, PNG export)
- **Sprint 11:** Scene Description Language (YAML/JSON)

## Backlog

Unscheduled items for future consideration:

| Item | Notes |
|------|-------|
| Caustics | Deferred - algorithm issues, branch `feature/caustics` preserved |
| Composites | Multiple overlapping objects |
| Coordinate cross | Visual axis reference |
| More primitives | Cylinders, cones, torus |
| 4D sponge in OptiX | Tesseract projection rendering |
| Real-time preview | Interactive low-quality mode |
| GPU instancing | Efficient repeated geometry |

See [arc42 Section 11](docs/arc42/11-risks-and-technical-debt.md) for technical debt tracking.
