# Fractional Level Implementation for 4D Sponges

**Date:** 2026-01-27
**Sprint:** 9 - TesseractSponge
**Status:** ✅ Complete

## Overview

Implemented proper fractional level support for 4D Menger sponges (tesseract-sponge and tesseract-sponge-2) in OptiX rendering. This enables smooth transitions between integer levels for animation purposes, matching the behavior of the LibGDX reference implementation.

## Problem Statement

Previously, fractional levels like `level=1.5` were accepted but rendered incorrectly:
- Only floored to integer level (1.5 → 1)
- No transparency overlay
- Result looked identical to `level=1.0`

This prevented smooth animation transitions between fractal levels.

## Solution

Implemented dual-object rendering approach:

### For `level=1.5`:
1. **Next level (opaque base):** `floor(1.5 + 1) = 2` with full opacity
2. **Current level (transparent overlay):** `floor(1.5) = 1` with `alpha = 1.0 - 0.5 = 0.5`

### Alpha Calculation
```scala
val fractionalPart = level - level.floor
val alpha = 1.0f - fractionalPart
```

This matches the LibGDX formula in `FractionalRotatedProjection.scala`.

## Implementation Details

### Modified Files

#### 1. `TriangleMeshSceneBuilder.scala`

**Key Methods:**

```scala
// Detects if a spec is a 4D sponge with fractional level
private def isFractional4DSponge(spec: ObjectSpec): Boolean =
  ObjectType.is4DSponge(spec.objectType) && spec.level.exists(isFractional)

// Groups specs by required base geometries
// Fractional levels create TWO geometry groups
private def groupByGeometry(specs: List[ObjectSpec]): Map[ObjectSpec, List[(ObjectSpec, Float)]]

// Counts instances (2 per fractional level)
override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
  specs.map { spec =>
    if isFractional4DSponge(spec) then 2L else 1L
  }.sum
```

**Build Scene Logic:**
- Groups specs by required geometries
- For fractional levels, creates entries for both `floor(level)` and `floor(level+1)`
- Sets base geometry once per group
- Adds instances with appropriate alpha values

**Compatibility:**
- 3D sponges: Still require matching levels (shared geometry)
- 4D sponges: Allow different levels (handled via multiple geometries)
- Mixed integer/fractional levels: Fully supported

#### 2. `FractionalLevelSceneBuilderSuite.scala` - NEW

Comprehensive test coverage (28 tests):

- **Fractional detection:** Correctly identifies fractional vs integer levels
- **Instance counting:** 1 for integer, 2 for fractional
- **Geometry grouping:** Proper split for fractional levels
- **Alpha calculation:** Matches formula for all fractional parts
- **Compatibility:** Allows mixing integer and fractional levels
- **Edge cases:** 0.0, 1.0, 2.0 treated as integer; 1.01, 1.99 as fractional
- **Validation:** Proper instance limit enforcement

### Test Results

All 1,302 tests pass:
- 28 new fractional level tests (100% pass rate)
- All existing tests still passing
- No regressions

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

Fractional levels create 2 instances instead of 1:
- Must account for this in instance count limits
- Validation now checks `calculateInstanceCount()` instead of `specs.length`

### Geometry Groups

The `groupByGeometry()` method creates a map:
```
Map[geomSpec -> List[(instanceSpec, alpha)]]
```

Where:
- `geomSpec`: Normalized spec with integer level for creating base geometry
- `instanceSpec`: Original spec for position/material
- `alpha`: Transparency value (1.0 for opaque, < 1.0 for transparent)

### Alpha Blending

Material alpha is set via:
```scala
val material = baseMaterial.copy(
  color = baseMaterial.color.copy(a = alpha)
)
```

OptiX handles the transparency rendering automatically.

## Performance Considerations

- **Instance count:** Fractional levels double instance count
  - Level 1.5: 2 instances vs 1 for level 1.0
  - Still much faster than generating intermediate geometry
- **Memory:** Slight increase due to dual geometry storage
- **Render time:** Minimal impact (transparency handled by GPU)

## Comparison with LibGDX

| Aspect | LibGDX | OptiX (This Implementation) |
|--------|--------|----------------------------|
| **Approach** | Dual `RotatedProjection` objects | Dual geometry groups |
| **Alpha formula** | `1.0 - fractionalPart` | ✅ Same |
| **Base level** | `floor(level + 1)` | ✅ Same |
| **Overlay level** | `floor(level)` | ✅ Same |
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
