# Fractional Level Implementation for 4D Sponges

**Date:** 2026-01-29
**Sprint:** 9 - TesseractSponge
**Status:** ✅ Complete (Per-Vertex Alpha Implementation)

## Overview

Implemented proper fractional level support for 4D Menger sponges (tesseract-sponge and tesseract-sponge-2) in OptiX rendering using **per-vertex alpha blending**. This enables smooth transitions between integer levels for animation purposes, matching the behavior of the LibGDX reference implementation.

## Problem Statement

Previously, fractional levels like `level=1.5` were broken:
- Attempted dual-instance rendering by calling `setTriangleMesh()` twice
- **Second call overwrites the first** (OptiX only supports ONE vertex buffer at a time)
- Result: Random pixel noise, no proper structure

This prevented smooth animation transitions between fractal levels.

## Solution

Implemented **per-vertex alpha with merged geometry**:

### For `level=1.5`:
1. Generate **level 2** mesh, assign per-vertex alpha = **1.0** (fully opaque)
2. Generate **level 1** mesh, assign per-vertex alpha = **0.5** (1.0 - fractionalPart)
3. **Merge both meshes** into single TriangleMeshData with stride=9
4. Upload as **one triangle mesh** to OptiX
5. Shader **interpolates vertex alpha** using barycentric coordinates
6. Final alpha: `vertex_alpha × material_alpha`

### Alpha Calculation
```scala
val fractionalPart = level - level.floor
val alphaTransparent = 1.0f - fractionalPart  // For level N
val alphaOpaque = 1.0f                         // For level N+1
```

This matches the LibGDX formula in `FractionalRotatedProjection.scala` but uses per-vertex alpha instead of dual objects.

## Implementation Details

### Modified Files

#### 1. `TriangleMeshData.scala` - Extended Vertex Format

**New stride=9 support (pos+normal+uv+alpha):**

```scala
// Extended vertex stride with per-vertex alpha
val VertexStrideWithAlpha: Int = 9

// Add alpha channel to stride=8 mesh
def withAlpha(mesh: TriangleMeshData, alpha: Float): TriangleMeshData =
  require(mesh.vertexStride == 8, "Can only add alpha to stride=8 meshes")
  require(alpha >= 0.0f && alpha <= 1.0f, "Alpha must be in [0.0, 1.0]")

  val newVertices = new Array[Float](mesh.numVertices * 9)
  for (i <- 0 until mesh.numVertices) {
    // Copy pos + normal + uv (8 floats)
    System.arraycopy(mesh.vertices, i * 8, newVertices, i * 9, 8)
    // Add alpha as 9th component
    newVertices(i * 9 + 8) = alpha
  }

  TriangleMeshData(newVertices, mesh.indices, vertexStride = 9)
```

#### 2. `TriangleMeshSceneBuilder.scala` - Fractional Mesh Creation

**Key Methods:**

```scala
// Detects if a spec is a 4D sponge with fractional level
private def isFractional4DSponge(spec: ObjectSpec): Boolean =
  ObjectType.is4DSponge(spec.objectType) && spec.level.exists(isFractional)

// Create merged mesh with per-vertex alpha for fractional levels
private def createFractionalMesh(spec: ObjectSpec): TriangleMeshData =
  val level = spec.level.get
  val fractionalPart = level - level.floor
  val alphaTransparent = 1.0f - fractionalPart

  // Generate both level geometries
  val nextLevelSpec = spec.copy(level = Some((level + 1).floor))
  val currentLevelSpec = spec.copy(level = Some(level.floor))

  val nextLevel = MeshFactory.create(nextLevelSpec)
  val currentLevel = MeshFactory.create(currentLevelSpec)

  // Assign per-vertex alpha
  val nextWithAlpha = TriangleMeshData.withAlpha(nextLevel, 1.0f)
  val currentWithAlpha = TriangleMeshData.withAlpha(currentLevel, alphaTransparent)

  // Merge into single mesh
  TriangleMeshData.merge(Seq(nextWithAlpha, currentWithAlpha))

// Instance count (1 per fractional level - merged mesh)
override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
  specs.map { spec =>
    if isFractional4DSponge(spec) then 1L else 1L  // All are 1 instance now
  }.sum
```

**Build Scene Logic:**
- Detects fractional levels in `buildScene()`
- Creates merged mesh via `createFractionalMesh()`
- Uploads as single triangle mesh
- Adds one instance per spec (fractional or not)

#### 3. `hit_triangle.cu` - Shader Alpha Interpolation

**Extended TriangleGeometry struct:**

```cuda
struct TriangleGeometry {
    float3 hit_point;
    float3 normal;
    float2 uv_coords;
    float t;
    bool entering;
    float vertex_alpha;  // NEW: Interpolated per-vertex alpha
};
```

**Per-vertex alpha interpolation:**

```cuda
// In getTriangleGeometry():
float vertex_alpha = 1.0f;
if (stride >= 9) {
    vertex_alpha = w * v0[8] + u * v1[8] + v * v2[8];
}
geom.vertex_alpha = vertex_alpha;
```

**Alpha multiplication:**

```cuda
// In getTriangleMaterial():
// Multiply material alpha with per-vertex alpha
color.w *= vertex_alpha;
```

#### 4. `JNIBindings.cpp` - Stride Validation

**Updated validation to accept stride=9:**

```cpp
// Validate vertex stride (6, 8, or 9)
if (vertexStride != 6 && vertexStride != 8 && vertexStride != 9) {
    jclass exception_class = env->FindClass("java/lang/IllegalArgumentException");
    env->ThrowNew(exception_class, "vertexStride must be 6, 8, or 9");
    return;
}
```

### Test Results

All 1,302 tests pass:
- All existing tests still passing
- No regressions
- Manual testing confirms proper fractional rendering (not random noise)

## Usage Examples

### Command Line

```bash
# Integer level (baseline)
--objects type=tesseract-sponge:level=1

# Fractional levels (smooth transition)
--objects type=tesseract-sponge:level=1.25  # 75% transparent overlay
--objects type=tesseract-sponge:level=1.5   # 50% transparent overlay
--objects type=tesseract-sponge:level=1.75  # 25% transparent overlay

# Multiple fractional levels in same scene
--objects type=tesseract-sponge:level=1.3:pos=-2,0,0 \
--objects type=tesseract-sponge:level=1.6:pos=0,0,0 \
--objects type=tesseract-sponge:level=1.9:pos=2,0,0

# Works with tesseract-sponge-2 too
--objects type=tesseract-sponge-2:level=2.5
```

### Manual Testing

Run the test script to generate images:

```bash
./scripts/test-fractional-levels.sh
```

Generates images for levels 1.0, 1.25, 1.5, 1.75, 2.0 showing smooth visual transition.

## Technical Notes

### Instance Budget

Fractional levels create **1 merged instance** (not 2):
- No change to instance count limits for fractional levels
- Geometry is larger (more triangles) but only one instance

### Vertex Format Extension

Extended from stride=8 to stride=9:
```
Stride 6: [px, py, pz, nx, ny, nz]
Stride 8: [px, py, pz, nx, ny, nz, u, v]
Stride 9: [px, py, pz, nx, ny, nz, u, v, alpha]  // NEW
```

### Barycentric Interpolation

Shader interpolates vertex alpha across triangle:
```cuda
float u = barycentrics.x;
float v = barycentrics.y;
float w = 1.0f - u - v;

vertex_alpha = w * v0[8] + u * v1[8] + v * v2[8];
```

### Alpha Blending

Final alpha is computed as:
```
final_alpha = vertex_alpha × material_alpha
```

Where:
- `vertex_alpha`: Interpolated from vertices (1.0 for level N+1, < 1.0 for level N)
- `material_alpha`: From material specification

OptiX handles the transparency rendering automatically.

## Performance Considerations

- **Instance count:** Fractional levels use **1 merged instance** (same as integer levels)
  - No increase in instance count
  - Instance limit unchanged
- **Triangle count:** Approximately doubled for fractional levels
  - Level 1.5: triangles from level 1 + triangles from level 2
  - Still acceptable for GPU rendering
- **Memory:** Increased vertex buffer size (stride=9 vs stride=8, plus merged geometry)
- **Render time:** Minimal impact (transparency handled by GPU, barycentric interpolation is fast)

## Comparison with LibGDX

| Aspect | LibGDX | OptiX (This Implementation) |
|--------|--------|----------------------------|
| **Approach** | Dual `RotatedProjection` objects | Merged mesh with per-vertex alpha |
| **Alpha formula** | `1.0 - fractionalPart` | ✅ Same |
| **Base level** | `floor(level + 1)` | ✅ Same (opaque, alpha=1.0) |
| **Overlay level** | `floor(level)` | ✅ Same (transparent, alpha=1.0-frac) |
| **Blending** | Material alpha | Per-vertex alpha × material alpha |
| **Result** | Smooth transition | ✅ Smooth transition |

## Future Enhancements

- [ ] Add fractional level support for 3D sponges (currently only 4D)
- [ ] Animated sequence generator using fractional levels
- [ ] Performance optimization for scenes with many fractional levels

## References

- **LibGDX reference:** `menger-app/src/main/scala/menger/objects/higher_d/FractionalRotatedProjection.scala`
- **Sprint documentation:** `docs/sprints/SPRINT9.md`
- **Test suite:** `menger-app/src/test/scala/menger/engines/scene/FractionalLevelSceneBuilderSuite.scala`
- **Manual test script:** `scripts/test-fractional-levels.sh`
