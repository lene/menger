# Sprint 33: PBR Texture Sets

**Sprint:** 33 - PBR Texture Sets
**Status:** Not Started
**Estimate:** ~24 hours
**Branch:** `feature/sprint-33`
**Dependencies:** None (builds on the Sprint 20/21 texture pipeline)
**Feature ID:** T-PBRTEX in [FEATURE_DEPENDENCIES.md](FEATURE_DEPENDENCIES.md)

---

## Goal

Load complete, published PBR texture sets (ambientCG, Poly Haven, and compatible CC0
libraries) by pointing at a folder — no per-map DSL plumbing. Requires adding the
missing material map slots (metallic, ambient occlusion, height-as-bump) next to the
existing albedo/normal/roughness maps. This is the groundwork MaterialX support
(TODO.md) will later reuse.

---

## Success Criteria

- [ ] `--objects 'type=sphere:texture-set=PavingStones070'` (folder under
      `--texture-dir`) auto-binds color, normal, roughness, metallic, and AO maps
- [ ] Both ambientCG and Poly Haven naming conventions are detected
- [ ] Explicit per-map parameters override auto-detected maps
- [ ] Metallic and AO maps visibly affect shading (new shader slots)
- [ ] A tiny CC0 texture set is committed for deterministic tests
- [ ] All tests pass

---

## Tasks

### Task 33.1: New Material Map Slots (Shader + InstanceMaterial)

**Estimate:** 8h

**Implementation:**
- `InstanceMaterial` gains texture indices for: metallic map, AO map, height map
  (existing: albedo/`texture`, normal map, roughness map)
- Shader sampling in closest-hit:
  - metallic map → per-texel metallic (multiplies the scalar `metallic` parameter,
    same pattern as roughness map)
  - AO map → multiplies the diffuse/ambient term only (not specular — standard
    PBR-workflow behavior)
  - height map → bump-derived normal perturbation via central differences of the
    height texture (only when no explicit normal map is present; a normal map wins).
    True displacement is out of scope — note for the displaced-micro-mesh item in
    the Sprint 30 API audit
- All slots optional; unset slots sample as identity (cost: one branch per slot,
  same as existing maps)
- UV handling identical to existing maps (sphere/cone/plane/mesh UVs from
  Sprints 20–21)

---

### Task 33.2: Texture-Set Loader (Naming Conventions)

**Estimate:** 6h

**Implementation:**
- `TextureSetResolver` in `menger-app` (pure Scala, unit-testable without GPU):
  given a directory, classify files by suffix patterns:
  - ambientCG: `*_Color.*`, `*_NormalGL.*`, `*_Roughness.*`, `*_Metalness.*`,
    `*_AmbientOcclusion.*`, `*_Displacement.*`
  - Poly Haven: `*_diff_*`, `*_nor_gl_*`, `*_rough_*`, `*_metal_*`, `*_ao_*`,
    `*_disp_*`
  - case-insensitive; `NormalDX` variants rejected with a clear error (DirectX-style
    normals need green-channel flip — implement the flip rather than reject if
    trivial in the existing loader)
- Resolution rules: explicit `texture=` / `normal-map=` / etc. parameters always
  override set-detected maps; multiple resolution variants in one folder (1K/2K/4K)
  → pick highest unless `texture-set-res=1K` given
- Result: a `ResolvedTextureSet(color, normal, roughness, metallic, ao, height)` of
  `Option[Path]`, fed through the existing `TextureManager` upload path into the new
  slots

---

### Task 33.3: Set Metadata (IOR, UV Scale)

**Estimate:** 3h

**Implementation:**
- Optional JSON sidecar `menger-textureset.json` in the set folder:
  `{ "ior": 1.45, "uvScale": 2.0 }` — covers the "sets can carry IOR" requirement
  without inventing a format for data the libraries don't ship; users add the file
  when needed
- `uvScale` multiplies UV coordinates for all maps of the set (tiling control —
  practically required for plane geometry)
- DSL: `TextureSet(path, uvScale = ..., ior = ...)` constructor args override the
  sidecar

---

### Task 33.4: CLI + DSL Wiring

**Estimate:** 3h

- CLI: `texture-set=NAME` (resolved against `--texture-dir`), `texture-set-res=`,
  `uv-scale=` in the `--objects` spec
- DSL: `SceneObject.textureSet: Option[TextureSet]`; mutually exclusive with
  `videoTexture`; combinable with `texture=`-style overrides per the resolution rules
- Validation errors name the offending file/folder explicitly

---

### Task 33.5: Tests + Reference Images + Documentation

**Estimate:** 4h

- Commit a tiny (64×64) CC0 texture set exercising all five maps; unit tests for
  `TextureSetResolver` (both conventions, override rules, resolution pick, DX-normal
  handling)
- Integration: textured sphere + plane reference images with the committed set;
  AO/metallic visual-difference assertions (with-map vs. without differ)
- `scripts/manual-test.sh`: one real-world-style set scene (append at end)
- User guide: PBR Texture Sets section (conventions table, sidecar format, override
  rules); CHANGELOG.md entry; note in TODO.md MaterialX entry that map slots now
  exist

---

## Summary

| Task | Description | Estimate |
|------|-------------|----------|
| 33.1 | Metallic/AO/height map slots in shaders | 8h |
| 33.2 | Texture-set resolver (naming conventions) | 6h |
| 33.3 | Set metadata (IOR, UV scale) | 3h |
| 33.4 | CLI + DSL wiring | 3h |
| 33.5 | Tests + reference images + documentation | 4h |
| **Total** | | **~24h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Integration + manual test scripts cover texture sets
