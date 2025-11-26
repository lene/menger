# Next Steps: Architectural Refactoring

**Date:** 2025-11-26
**Branch:** `feature/caustics`
**Previous Work:** Completed "Stone 1" - GeometryFactory extraction and Observer decoupling (issues 4.1 + 4.2)

---

## Completed: Stone 1 ✅

**Issues Resolved:**
- **4.1** - Domain/UI Coupling: Removed Observer from Geometry trait
- **4.2** - Factory Logic in Wrong Place: Extracted GeometryFactory from MengerEngine

**Results:**
- MengerEngine reduced from ~82 to 38 lines
- Clean separation of domain model (3D geometries) from UI concerns (events)
- All 85 tests passing
- Committed: `4560eb5` - "refactor: Extract GeometryFactory and decouple Observer from Geometry trait"

---

## Remaining Architectural Inadequacies

Grouped by potential synergy for efficient refactoring:

### **Stone 2** - Liskov Substitution Principle & God Object (HIGH IMPACT)
**Synergy:** OptiXEngine violates LSP because it inherits from MengerEngine but doesn't use many features. This is related to OptiXResources having too many responsibilities. Fixing both together would create cleaner architecture.

#### 1.3 - OptiXEngine Violates LSP
- **File:** `src/main/scala/menger/engines/OptiXEngine.scala:72-76`
- **Problem:**
  ```scala
  protected def drawables: List[ModelInstance] =
    throw new UnsupportedOperationException("OptiXEngine doesn't use drawables")

  protected def gdxResources: GDXResources =
    throw new UnsupportedOperationException("OptiXEngine doesn't use gdxResources")
  ```
- **Impact:** Violates Liskov Substitution Principle - child class throws exceptions for parent methods
- **Fix Options:**
  1. Create `RenderEngine` trait with minimal contract
  2. Use composition instead of inheritance
  3. Make MengerEngine methods optional (Option[GDXResources])

#### 4.3 - OptiXResources Has Too Many Responsibilities (SRP Violation)
- **File:** `src/main/scala/menger/OptiXResources.scala` (175 lines)
- **Current Responsibilities:**
  - JNI initialization
  - Camera management
  - Light configuration
  - Plane configuration
  - Statistics reporting
  - Camera updates
- **Suggested Split:**
  - `SceneConfigurator` - lights, plane, camera initial setup
  - `OptiXRendererWrapper` - JNI bridge, render calls
  - `CameraState` - camera position/direction updates

**Combined Approach for Stone 2:**
1. Define minimal `RenderEngine` interface (or use composition)
2. Extract OptiXResources responsibilities into focused classes
3. Update OptiXEngine to compose rather than inherit
4. Estimated effort: 4-6 hours

---

### **Stone 3** - Large Files/Functions Decomposition (MAINTAINABILITY)
**Synergy:** All related to breaking large units into smaller, focused, testable components.

#### 4.4 - Oversized Files
| File | Lines | Threshold | Issue |
|------|-------|-----------|-------|
| `sphere_combined.cu` | 1700 | 500 | 26 shader programs, PPM caustics |
| `OptiXWrapper.cpp` | 1080 | 500 | Monolithic, 50+ member variables |
| `MengerCLIOptions.scala` | 473 | 300 | All CLI parsing in one file |

#### 4.5 - Oversized Functions
| Function | File | Lines | Max |
|----------|------|-------|-----|
| `render()` | OptiXWrapper.cpp:582-805 | 223 | 50 |
| `dispose()` | OptiXWrapper.cpp:950-1080 | 130 | 50 |
| `__closesthit__ch()` | sphere_combined.cu:687-947 | 260 | 50 |

#### 5.1 - Shader File Should Be Split
**File:** `sphere_combined.cu` (1700 lines)

**Recommended Split:**
- `raygen_primary.cu` - Primary camera ray generation
- `hit_sphere.cu` - Sphere material (Fresnel, refraction, Beer-Lambert)
- `miss_plane.cu` - Plane rendering and lighting
- `shadows.cu` - Shadow ray tracing with transparency
- `caustics_ppm.cu` - Progressive Photon Mapping (~730 lines)

**Combined Approach for Stone 3:**
1. Split `sphere_combined.cu` into 5 focused shader files
2. Extract helper functions from `render()` and `dispose()` in OptiXWrapper.cpp
3. Consider splitting MengerCLIOptions by feature (rendering, caustics, lights, etc.)
4. Estimated effort: 6-8 hours

---

### **Stone 4** - C++ Code Quality (RELIABILITY)
**Synergy:** Both improve C++ code robustness and completeness.

#### 5.2 - Missing Error Recovery in Buffer Allocation
- **File:** `OptiXWrapper.cpp:629-688`
- **Problem:** If 3rd of 7 `cudaMalloc` calls fails, first 2 buffers leak
- **Fix:** RAII wrapper or cleanup-on-failure pattern
- **Estimated effort:** 2 hours

#### 5.3 - Incomplete TODOs in Caustics
| Location | TODO | Status |
|----------|------|--------|
| `sphere_combined.cu:1241` | Use spatial hash grid for efficiency | Not implemented |
| `sphere_combined.cu:1495` | Weight by intensity for multiple lights | Not implemented |

- **Estimated effort:** 3-4 hours (spatial hash grid is non-trivial)

**Combined Approach for Stone 4:**
1. Implement RAII buffer wrapper or cleanup pattern
2. Evaluate spatial hash grid benefits (profile first)
3. Implement light intensity weighting for caustics
4. Estimated effort: 5-6 hours

---

### **Lower Priority Issues** (Not Grouped)

#### 6.1 - Magic Numbers in Tests
**High-frequency values not using constants:**
- `1.5f` (IOR glass) - 23+ occurrences
- `0.5f` (sphere radius) - 30+ occurrences
- `60.0f` (FOV) - 15+ occurrences

**Fix:** Create `MaterialConstants.scala` in test utilities
**Effort:** 1 hour

#### 6.2 - Inconsistent Test Patterns
- `WindowResizeTest` doesn't use `RendererFixture` trait
- `setupShadowScene()` helper only used in ~38% of shadow tests
- Test file naming inconsistent: `*Test.scala`, `*Spec.scala`, `*Suite.scala`

**Fix:** Standardize on AnyFlatSpec pattern, consistent naming
**Effort:** 2 hours

#### 6.3 - Mutable State in Input Controllers (FP Violation)
**Files:**
- `OptiXCameraController.scala:40-70` - 10 `var` fields
- `KeyController.scala:13-20` - 4 `var` fields

**Stated Rule:** "No `var`" per CLAUDE.md
**Current State:** 16+ mutable fields with `@SuppressWarnings` annotations

**Fix:** Use immutable state with functional updates (StateT monad or similar)
**Effort:** 3-4 hours
**Note:** May be acceptable to defer - input controllers are inherently stateful

---

## Recommended Order

1. **Stone 2** (LSP + OptiXResources) - Highest architectural impact
   - Improves separation of concerns
   - Makes OptiX integration cleaner
   - Enables future alternative renderers
   - Effort: 4-6 hours

2. **Stone 4** (C++ reliability) - Important for production
   - Prevents resource leaks
   - Completes existing features
   - Effort: 5-6 hours

3. **Stone 3** (File decomposition) - Improves maintainability
   - Makes codebase easier to navigate
   - Reduces merge conflicts
   - Better for future contributors
   - Effort: 6-8 hours

4. **Lower priority items** - Polish
   - Test constants (6.1): 1 hour
   - Test patterns (6.2): 2 hours
   - Input controller vars (6.3): 3-4 hours (optional)

---

## Total Remaining Effort Estimate

- **High Priority (Stones 2-4):** 15-20 hours
- **Polish (Lower priority):** 6-7 hours
- **Grand Total:** 21-27 hours

---

## Notes for Next Session

- Current branch has uncommitted changes (OptiXWrapper.cpp, test files from previous session)
- May want to commit those separately before starting next stone
- GeometryFactory refactoring provides good pattern for future extractions
- All current tests (818 total) are passing
