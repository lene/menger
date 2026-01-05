# Code Quality Improvement Opportunities

**Date**: 2026-01-05
**Status**: Pre-merge cleanup completed
**Purpose**: Track remaining improvement opportunities

---

## Remaining Work

### Medium Priority

| ID | Description | Est. Time |
|----|-------------|-----------|
| #46 | Extract Transform Matrix Utilities to Common Module | 4 hours |
| #48 | Implement Builder Pattern for ObjectSpec | 3-4 hours |
| #49 | Add Comprehensive Error Context | 5-6 hours |
| #33 | Extract remaining regex patterns | 1 hour |
| #34 | Add builder pattern for complex objects | 2 hours |
| #37 | Extract animation parameter parsing | 1.5 hours |
| #39 | Add more specific exception types | 2 hours |
| #40 | Improve test organization | 2 hours |
| #41 | Add property-based tests | 3 hours |
| #42 | Reduce cognitive complexity | 2 hours |
| #43 | Add debug logging strategically | 1 hour |

### Low Priority

| ID | Description | Est. Time |
|----|-------------|-----------|
| #50 | Configuration DSL | 8-10 hours |
| #51 | Metrics and Telemetry | 6-8 hours |
| #55 | Implement scene graph abstraction | 10-12 hours |
| #56 | Add comprehensive benchmarking suite | 8-10 hours |
| #57 | Create plugin system for geometry types | 12-15 hours |
| #58 | Add hot-reload for development | 6-8 hours |

---

## Documentation Quality Issues

### DOC-1. Missing API Documentation (High Priority)

**Files**:
- `ObjectSpec.scala` - Missing examples for all parse error cases
- `Transform utilities` - No documentation on matrix format

### DOC-2. Outdated TODO Comments (Low Priority)

**File**: `optix-jni/src/main/native/shaders/caustics_ppm.cu`

```cpp
// Line 287: TODO: Use spatial hash grid for efficiency
// Line 574: TODO: Weight by intensity for multiple lights
```

### DOC-3. Complex Algorithms Lack Explanation (Medium Priority)

**Files**:
- `CubeSpongeGenerator.scala` - How does recursive subdivision work?
- `SpongeBySurface.scala` - Face generation algorithm
- `caustics_ppm.cu` - Photon mapping algorithm

---

## C++ Code Quality Issues

### CPP-1. Magic Numbers in Shaders (Medium Priority)

**Files**: `caustics_ppm.cu`, `sphere_combined.cu`, `helpers.cu`

**Examples**:
- Ray offset values (0.001f, 0.0001f)
- Maximum ray depth (10, 20)
- Grid sizes (64, 128)

### CPP-2. Long Functions in Shaders (Low Priority)

Some shader functions exceed 50 lines. Consider breaking them down for readability.

---

## Testing Quality Issues

### TEST-2. Insufficient Edge Case Testing (Low Priority)

Few tests cover:
- Empty inputs
- Boundary conditions (max values)
- Invalid combinations

---

## Deferred / Explicitly Accepted Issues

The following issues were explicitly deferred or accepted:

1. **Mutable state in LibGDX integration** - Required by framework, properly suppressed
2. **OptiX cache management** - Works correctly, no changes needed
3. **Caustics algorithm issues** - Deferred to future sprint, not blocking
4. **Test performance** - Acceptable, optimization not priority

---

## Estimated Total Effort

- **Medium priority items**: ~24 hours
- **Low priority items**: ~51 hours
- **Documentation/Testing**: ~8 hours

**Total remaining**: ~83 hours (10.5 days)

---

## Notes

1. **Already Compliant**: No `var` or `throw` in production code
2. **Good Practices**: Functional style, wartremover enforcement, scalafix integration
3. **Sprint 6 completed**: 28 issues addressed, critical refactoring done
