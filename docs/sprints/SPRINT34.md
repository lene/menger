# Sprint 34: PBR Texture Sets

**Sprint:** 34 - PBR Texture Sets
**Status:** 📋 Planned — scope confirmed 2026-07-02
**Estimate:** ~24 hours
**Branch:** `feature/sprint-34` (created)
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

## Research Summary

### Poly Haven convention (`name_suffix.ext`)

| Suffix | Map |
|--------|-----|
| `_diff` | Albedo / diffuse |
| `_rough` | Roughness |
| `_metal` | Metallic |
| `_nor_gl` | Normal (OpenGL, green=up) |
| `_nor_dx` | Normal (DirectX, green=down — convert to OpenGL) |
| `_disp` | Displacement / height |
| `_ao` | Ambient Occlusion |

**No resolution in filename** — Poly Haven keeps resolution in folder structure
(e.g., `textures/4K/`, `textures/2K/`, `textures/1K/`).

### ambientCG convention (`name_map_4K.ext`)

| Suffix | Map |
|--------|-----|
| `_Color` | Albedo |
| `_NormalGL` | Normal (OpenGL) |
| `_NormalDX` | Normal (DirectX) |
| `_Roughness` | Roughness |
| `_Metalness` | Metallic |
| `_AmbientOcclusion` | AO |
| `_Displacement` | Height |
| `_Opacity` | Alpha mask |

**Resolution in filename** — e.g., `PavingStones067_Color_4K.jpg`.
Multiple resolutions in flat folder.

### State of the existing pipeline

**Current GPU slots** (`InstanceMaterial` in `OptiXData.h`):
- `image_texture_index` (albedo) ✅
- `normal_texture_index` ✅
- `roughness_texture_index` ✅

**Current ObjectSpec fields:**
- `texture: Option[String]` (albedo)
- `textureMaps.normalMap: Option[String]`
- `textureMaps.roughnessMap: Option[String]`

**Missing (Sprint 34 adds):**
- `metallic_texture_index` (+ shader accessor `getInstanceMetallicTextureIndex`)
- `ao_texture_index` (+ shader accessor)
- `height_texture_index` (+ shader accessor)
- New ObjectSpec fields: `metallicMap`, `aoMap`, `heightMap`, `textureSet`, `textureSetRes`, `uvScale`

**Shader sampling:** `hit_sphere.cu` line 57-59 shows the pattern:
```c
material_color = applyImageTexture(material_color, sphere_uv, getInstanceImageTextureIndex());
geometric_normal = applyNormalMap(geometric_normal, sphere_uv, getInstanceNormalTextureIndex());
roughness = applyRoughnessMap(roughness, sphere_uv, getInstanceRoughnessTextureIndex());
```
New slots follow the same pattern.

### NormalDX → OpenGL conversion

DirectX normals have green channel inverted (Y-down vs Y-up). Conversion: flip green:
```c
float3 normal_dx = make_float3(sample.x, sample.y, sample.z);
normal_out = make_float3(normal_dx.x, 1.0f - normal_dx.y, normal_dx.z);
```


---

## Tasks

### Task 34.1: New Material Map Slots (Shader + InstanceMaterial)

**Estimate:** 8h
**Files touched:** `optix-jni/src/main/native/include/OptiXData.h` (InstanceMaterial struct),
  `optix-jni/src/main/native/shaders/helpers.cu` (accessors + apply functions),
  `optix-jni/src/main/native/shaders/hit_sphere.cu` (closest-hit integration),
  all other hit shaders (hit_plane.cu, hit_menger4d.cu, etc.)

**Step 1 — InstanceMaterial (OptiXData.h):**
Add three texture index fields after `roughness_texture_index`:
```c
int metallic_texture_index;  // -1 = no metallic map
int ao_texture_index;        // -1 = no AO map
int height_texture_index;    // -1 = no height/bump map
```
Update the struct on both the C++ side (`optix-jni`) and the copied version in menger-geometry.

**Step 2 — Shader accessors (helpers.cu):**
Add three new `__device__` accessors matching existing pattern:
```c
__device__ int getInstanceMetallicTextureIndex() { ... }
__device__ int getInstanceAoTextureIndex() { ... }
__device__ int getInstanceHeightTextureIndex() { ... }
```

**Step 3 — Material apply functions (helpers.cu):**
```c
__device__ float applyMetallicMap(float base_metallic, float2 uv, int tex_index) {
    if (tex_index < 0) return base_metallic;
    float4 sample = tex2D<float4>(params.textures[tex_index], uv.x, uv.y);
    return base_metallic * sample.x; // R channel, same as roughness map
}

__device__ float3 applyAOMap(float3 color, float2 uv, int tex_index) {
    if (tex_index < 0) return color;
    float4 sample = tex2D<float4>(params.textures[tex_index], uv.x, uv.y);
    return color * sample.x; // Multiply diffuse/ambient only
}
```

**Step 4 — Height-to-bump conversion (helpers.cu):**
When height map is present AND normal map is absent:
```c
__device__ float3 bumpFromHeight(float3 geometric_normal, float3 tangent,
                                  float3 bitangent, float2 uv, int tex_index,
                                  float bump_strength = 1.0f) {
    if (tex_index < 0) return geometric_normal;
    // Central differences: sample height ±1 texel, compute dp/du, dp/dv
    float eps = 1.0f / 512.0f; // texel size — use actual texture dimensions
    float h_l = tex2D<float4>(params.textures[tex_index], uv.x - eps, uv.y).x;
    float h_r = tex2D<float4>(params.textures[tex_index], uv.x + eps, uv.y).x;
    float h_d = tex2D<float4>(params.textures[tex_index], uv.x, uv.y - eps).x;
    float h_u = tex2D<float4>(params.textures[tex_index], uv.x, uv.y + eps).x;
    float3 dpdu = tangent * (h_r - h_l) * bump_strength;
    float3 dpdv = bitangent * (h_u - h_d) * bump_strength;
    return normalize(geometric_normal - dpdu - dpdv);
}
```

**Step 5 — Closest-hit integration (all hit shaders):**
After the existing roughness map apply (line 59), add:
```c
metallic = applyMetallicMap(metallic, sphere_uv, getInstanceMetallicTextureIndex());
material_color = applyAOMap(material_color, sphere_uv, getInstanceAoTextureIndex());
// Height bump: only when no normal map was applied
if (getInstanceNormalTextureIndex() < 0 && getInstanceHeightTextureIndex() >= 0) {
    geometric_normal = bumpFromHeight(geometric_normal, tangent, bitangent,
                                       sphere_uv, getInstanceHeightTextureIndex());
}
```
Update ALL hit shaders: hit_sphere.cu, hit_plane.cu, hit_menger4d.cu,
hit_sierpinski4d.cu, hit_hexadecachoron4d.cu, and the triangle mesh hit shader.

**Step 6 — JNI bridge:**
Update `addSphereInstance` and all `add*Instance` JNI functions to pass
`metallic_texture_index`, `ao_texture_index`, `height_texture_index`.
Also update `OptiXWrapper::addPlaneInstance`, `addConeInstance`,
`addCylinderInstance`, `addCurveInstance`, and the IAS-based
`addInstanceWithMaterial`.

**Step 7 — Scala API:**
Update `OptiXRenderer` `@native` declarations and `OptiXSphereApi`/
`OptiXPlaneApi` wrapper methods to accept Optional texture indices
(defaulting to -1).

**Step 8 — Point (custom GAS) geometry:**
The scalar `texture_index` for point geometry (line 443 of OptiXData.h)
is unchanged — points use only albedo texture.

**Gate:** `sbt nativeCompile` + all existing texture tests pass before proceeding.
Bump optix-jni to 0.1.11, publish to Maven Central.

---

### Task 34.2: Texture-Set Resolver (Naming Conventions)

**Estimate:** 6h
**Files:** New file `menger-app/src/main/scala/menger/engines/scene/TextureSetResolver.scala`,
  test file `TextureSetResolverSuite.scala`

**Implementation:**
`TextureSetResolver` — pure Scala, unit-testable without GPU. Given a directory
path, classify files by suffix patterns:

```scala
case class ResolvedTextureSet(
  color: Option[Path],
  normal: Option[Path],
  roughness: Option[Path],
  metallic: Option[Path],
  ao: Option[Path],
  height: Option[Path]
)
```

**Poly Haven detection** (lowercase, underscores):
| Map | Patterns |
|-----|----------|
| Albedo | `*_diff.*` |
| Normal | `*_nor_gl.*` (accept), `*_nor_dx.*` (convert) |
| Roughness | `*_rough.*` |
| Metallic | `*_metal.*` |
| AO | `*_ao.*` |
| Height | `*_disp.*` |

**ambientCG detection** (PascalCase, no underscores before map name):
| Map | Patterns |
|-----|----------|
| Albedo | `*_Color.*` |
| Normal | `*_NormalGL.*` (accept), `*_NormalDX.*` (convert) |
| Roughness | `*_Roughness.*` |
| Metallic | `*_Metalness.*` |
| AO | `*_AmbientOcclusion.*` |
| Height | `*_Displacement.*` |

**Detection algorithm:**
1. Scan directory for image files (`.png`, `.jpg`, `.jpeg`, `.tga`, `.hdr`)
2. For each file, test against Poly Haven patterns, then ambientCG patterns
3. First convention that matches at least 3 file types wins (detect-convention
   heuristic — prevents mixed-convention false matches)
4. Case-insensitive matching on filename
5. `nor_dx`/`NormalDX` files: record as normal with a `requiresConversion` flag

**Resolution rules:**
- Poly Haven: subdirectories `1K/`, `2K/`, `4K/`, etc. — scan all, pick highest
  resolution where ALL required maps exist
- ambientCG: resolution suffix in filename (`*_4K.jpg`) — extract, pick highest
- `texture-set-res=2K` CLI/DSL flag: forces specific resolution
- Multiple resolutions of same map type → highest wins (unless overridden)

**NormalDX → OpenGL conversion:**
When the only available normal map is DX-format, load the image and flip the
green channel during upload. This is done in `TextureManager` at upload time
(after detecting the DX flag from `TextureSetResolver`), not in the shader —
keeps the GPU path simple.

**Gate:** `TextureSetResolverSuite` — 15+ unit tests covering both conventions,
resolution picking, DX→GL conversion, empty directories, missing maps, override
behavior.

---

### Task 34.3: Set Metadata (IOR, UV Scale)

**Estimate:** 3h
**Files:** New file `menger-app/src/main/scala/menger/engines/scene/TextureSetMetadata.scala`

**Optional JSON sidecar** `menger-textureset.json` in the set folder:
```json
{
  "ior": 1.45,
  "uvScale": 2.0
}
```
All fields optional; missing fields = no override.

**Fallback chain:**
1. Explicit CLI parameter (`ior=`, `uv-scale=`) → always wins
2. JSON sidecar
3. material preset default

**Implementation:**
- `TextureSetMetadata.load(directory: Path): Try[TextureSetMetadata]`
- Reads `menger-textureset.json` if present, returns defaults otherwise
- Validation: ior must be ≥ 1.0, uvScale must be > 0

**Gate:** Metadata unit tests (valid JSON, missing file, invalid values).

---

### Task 34.4: CLI + DSL Wiring

**Estimate:** 3h
**Files:** `ObjectSpec.scala` (new fields + parsing), `MaterialExtractor.scala` (wiring),
  `TextureManager.scala` (loader integration)

**New ObjectSpec fields:**
```scala
metallicMap: Option[String] = None,
aoMap: Option[String] = None,
heightMap: Option[String] = None,
textureSet: Option[String] = None,    // folder name
textureSetRes: Option[String] = None,  // "1K" | "2K" | "4K"
uvScale: Option[Float] = None
```

**New ValidKeys entries:** `"metallic-map"`, `"ao-map"`, `"height-map"`,
`"texture-set"`, `"texture-set-res"`, `"uv-scale"`

**New parse functions:** `parseMetallicMap`, `parseAOMap`, `parseHeightMap`,
`parseTextureSet`, `parseTextureSetRes`, `parseUVScale`

**CLI syntax:**
```
--objects type=sphere:texture-set=PavingStones070:texture-set-res=2K:uv-scale=2.0
```
With explicit overrides:
```
--objects type=sphere:texture-set=PavingStones070:metallic-map=custom_metal.png
```

**TextureManager integration:**
When `spec.textureSet.isDefined`:
1. Resolve set directory: `$textureDir / textureSet`
2. Call `TextureSetResolver.resolve(setDir, spec.textureSetRes)` → `ResolvedTextureSet`
3. Load each resolved map through the existing upload path
4. Populate `textureIndices` with albedo/normal/roughness/metallic/ao/height indices
5. Explicit per-map params (`texture=`, `normal-map=`) override auto-detected

**Mutual exclusion:** `textureSet` is incompatible with `videoTexture`; combinable
with explicit `texture=`/`normal-map=` overrides.

**Validation errors:** Name the offending file/folder explicitly — e.g.,
"Texture set 'PavingStones070': no color/albedo map found (expected *_Color.* or *_diff.*)"

**Gate:** `sbt compile` + binary smoke test: `--objects type=sphere:texture-set=TINY_TEST_SET`.

---

### Task 34.5: Tests + Reference Images + Documentation

**Estimate:** 4h

**34.5a — Committed test set (1h):**
Create `scripts/test-assets/texture-sets/tiny-pbr/` with 64×64 PNGs:
- `tiny_diff.png` (checker pattern, red/white)
- `tiny_nor_gl.png` (flat normal = 128,128,255)
- `tiny_rough.png` (50% gray = 128)
- `tiny_metal.png` (white center, black edges)
- `tiny_ao.png` (white center, black corners)
- `menger-textureset.json`: `{"ior": 1.5, "uvScale": 1.0}`
All images CC0 (hand-authored noise patterns).

**34.5b — Unit tests (1h):**
- `TextureSetResolverSuite`: ~15 tests covering Poly Haven + ambientCG
  detection, resolution selection, DX→GL conversion flag, empty/broken sets,
  override precedence, case-insensitive matching
- `TextureSetMetadataSuite`: valid JSON, missing file defaults, invalid values
- `ObjectSpec` parse tests for new keys

**34.5c — Integration tests (1h):**
- Reference images for `texture-set=tiny-pbr` sphere + plane
- Diff-assertion: render with vs without metallic map → images differ (metalness
  changes specular appearance)
- AO diff-assertion: with vs without AO → corners darker
- Performance: textured sphere with all 5 maps < 100ms at 320×240

**34.5d — Documentation (1h):**
- `scripts/manual-test.sh`: PBR texture set entry (append at end)
- User guide: PBR Texture Sets section with conventions table, sidecar format,
  resolution rules, override precedence
- CHANGELOG.md entry
- Update `TODO.md` MaterialX entry: note that map slots now exist

---

## Task Dependency Graph

```
34.1 (new map slots) ────── must go first (optix-jni publish)
  │
  ├─► 34.2 (texture-set resolver) ──► 34.4 (CLI/DSL wiring)
  │                                      │
  └─► 34.3 (set metadata) ──────────────┤
                                         │
                                         ▼
                                    34.5 (tests/docs)
```

34.1 is a hard dependency — the GPU texture slots must exist before anything
else can upload to them. 34.2 and 34.3 are independent (both pure Scala) and
can be developed in parallel after 34.1 publishes. 34.4 wires them together.
34.5 is the final integration + documentation pass.

---

## Summary

| Task | Description | Est |
|------|-------------|-----|
| 34.1 | Metallic/AO/height map slots (GPU + JNI + Scala API) | 8h |
| 34.2 | Texture-set resolver (naming conventions) | 6h |
| 34.3 | Set metadata sidecar (IOR, UV scale) | 3h |
| 34.4 | CLI + DSL wiring | 3h |
| 34.5 | Tests + reference images + documentation | 4h |
| **Total** | | **~24h** |

---

## Definition of Done

- [ ] All success criteria met
- [ ] Pre-push hook green
- [ ] CHANGELOG.md updated
- [ ] Integration + manual test scripts cover texture sets
- [ ] optix-jni published with new texture slots (0.1.11+)
- [ ] `TextureSetResolverSuite` covers both conventions, edge cases
