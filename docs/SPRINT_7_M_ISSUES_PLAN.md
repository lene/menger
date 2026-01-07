# Sprint 7: Medium Priority Issues Implementation Plan

**Created:** 2026-01-07
**Total Estimated Hours:** ~21h

## Summary

| Issue | Status | Hours |
|-------|--------|-------|
| ~~M1~~ | **CLOSED** - Already complete | 0 |
| M4 | Extract regex patterns | 1 |
| M7 | Add specific exception types | 2 |
| M2 | Helper methods for Material | 3 |
| M5 | Review other config classes | 0.5 |
| M3 | Comprehensive error context | 5-6 |
| M10 | Strategic debug logging | 1 |
| M6 | Extract animation parsing | 1.5 |
| M9 | Reduce cognitive complexity | 2 |
| M11 | Encapsulate mutable state | 4 |
| **Total** | | **~21h** |

---

## Execution Order

```
Phase 1: Foundations (3h)
├── M4: Extract regex patterns (1h)
└── M7: Add exception types (2h)

Phase 2: Builder Patterns (3.5h)
├── M2: Material helpers (3h)
└── M5: Review config classes (0.5h)

Phase 3: Error Handling (6-7h)
├── M3: Error context (5-6h)
└── M10: Debug logging (1h)

Phase 4: Refactoring (3.5h)
├── M6: Animation parsing (1.5h)
└── M9: Cognitive complexity (2h)

Phase 5: Framework-Constrained (4h)
└── M11: Mutable state (4h)
```

---

## Detailed Implementation Plan

### 1. M4: Extract Remaining Regex Patterns (1h)

**Problem:** Duplicate `compositePattern` regex in two files.

**Files to modify:**
- `src/main/scala/menger/objects/Composite.scala` (has regex)
- `src/main/scala/menger/MengerCLIOptions.scala` (has duplicate regex)

**Actions:**
1. Create `menger.common.Patterns` object with shared regex patterns
2. Move `compositePattern` to the shared location
3. Update both files to import from common location
4. Verify tests pass

---

### 2. M7: Add Specific Exception Types (2h)

**Problem:** Generic `IllegalArgumentException` used everywhere.

**Files to modify:**
- Create new: `menger-common/src/main/scala/menger/common/MengerExceptions.scala`
- Update: `Direction.scala`, `AnimationSpecificationSequence.scala`

**Exception hierarchy:**
```scala
// Base exception for all Menger errors
sealed abstract class MengerException(message: String, cause: Option[Throwable] = None)
  extends RuntimeException(message, cause.orNull)

// Parsing errors (invalid input format)
case class ParseException(message: String, input: String, cause: Option[Throwable] = None)
  extends MengerException(s"$message. Input: '$input'", cause)

// Validation errors (valid format, invalid values)
case class ValidationException(message: String, field: String, value: Any)
  extends MengerException(s"$message. Field: $field, Value: $value")

// Configuration errors
case class ConfigurationException(message: String, cause: Option[Throwable] = None)
  extends MengerException(message, cause)
```

---

### 3. M2: Helper Methods for Material (3h)

**Problem:** Verbose copy chain in `ObjectSpec.parseMaterial`.

**Files to modify:**
- `optix-jni/src/main/scala/menger/optix/Material.scala` (add helpers)
- `src/main/scala/menger/ObjectSpec.scala` (use helpers)

**Add to Material class:**
```scala
case class Material(...):
  def withColorOpt(c: Option[Color]): Material = c.fold(this)(v => copy(color = v))
  def withIorOpt(i: Option[Float]): Material = i.fold(this)(v => copy(ior = v))
  def withRoughnessOpt(r: Option[Float]): Material = r.fold(this)(v => copy(roughness = v))
  def withMetallicOpt(m: Option[Float]): Material = m.fold(this)(v => copy(metallic = v))
  def withSpecularOpt(s: Option[Float]): Material = s.fold(this)(v => copy(specular = v))
```

**Refactor ObjectSpec.parseMaterial to:**
```scala
Some(
  baseMaterial
    .withColorOpt(color)
    .withIorOpt(Option.when(kvPairs.contains("ior"))(ior))
    .withRoughnessOpt(kvPairs.get("roughness").map(_.toFloat))
    .withMetallicOpt(kvPairs.get("metallic").map(_.toFloat))
    .withSpecularOpt(kvPairs.get("specular").map(_.toFloat))
)
```

---

### 4. M5: Review Other Config Classes (0.5h)

**Status:** Already well-structured with factory methods.

**Action:** Quick review of SceneConfig, OptiXEngineConfig, CameraConfig, etc. to confirm no changes needed. Document as complete if patterns are already clean.

---

### 5. M3: Comprehensive Error Context (5-6h)

**Problem:** Error messages lack context for debugging.

**Files to modify:**
- `ObjectSpec.scala` - enhance parse error messages
- `AnimationSpecification.scala` - add context to validation errors
- `cli/converters/*.scala` - improve conversion error messages
- Various others using M7's exception types

**Pattern to follow:**
```scala
// Before
Left(s"Invalid value: $value")

// After
Left(ParseException(
  message = "Invalid color format",
  input = value,
  hint = "Expected hex format (#RRGGBB) or named color"
).getMessage)
```

---

### 6. M10: Strategic Debug Logging (1h)

**Files to modify:**
- `ObjectSpec.scala` - add debug logging for parse steps
- `AnimationSpecification.scala` - log validation
- `CliValidation.scala` - log validation rule execution

**Pattern:**
```scala
logger.debug(s"Parsing object spec: $input")
logger.debug(s"Parsed key-value pairs: $kvPairs")
```

---

### 7. M6: Extract Animation Parameter Parsing (1.5h)

**Problem:** Animation parsing scattered across files.

**Files to review:**
- `MengerCLIOptions.scala`
- `AnimationSpecification.scala`
- `cli/converters/SceneConverters.scala`

**Action:** Ensure clear separation - parsing in one place, validation in another.

---

### 8. M9: Reduce Cognitive Complexity (2h)

**Files to review:**
- `OptiXEngine.scala` - already well-factored, may need minor extraction
- `CliValidation.scala` - `registerValidationRules` is long but organized
- `ObjectSpec.scala` - will be improved by M2

**Action:** Identify methods >20 lines, extract where it improves clarity.

---

### 9. M11: Encapsulate Mutable State (4h)

**Problem:** Multiple `var` declarations in input controllers.

**Files to modify:**
- `input/BaseKeyController.scala` (4 vars)
- `input/OptiXCameraController.scala` (10 vars)

**Approach:**
```scala
// Encapsulate keyboard state
case class KeyboardState(
  ctrlPressed: Boolean = false,
  altPressed: Boolean = false,
  shiftPressed: Boolean = false,
  rotatePressed: Map[Int, Boolean] = Map().withDefaultValue(false)
)

// Encapsulate camera state  
case class CameraState(
  azimuth: Float = 0f,
  elevation: Float = 0f,
  distance: Float = 5f,
  // ... other fields
)

// Single var instead of many
@SuppressWarnings(Array("org.wartremover.warts.Var"))
private var keyState: KeyboardState = KeyboardState()
```

**Revert condition:** If this doesn't provide meaningful improvement (cleaner code, easier testing), revert and close as "accepted framework constraint."

---

## Commits Plan

| Commit | Issues | Message |
|--------|--------|---------|
| 1 | M1 | `docs: Close M1 - TransformUtil already complete` |
| 2 | M4 | `refactor: Extract shared regex patterns to common module (M4)` |
| 3 | M7 | `feat: Add specific exception types (M7)` |
| 4 | M2, M5 | `refactor: Add Material helper methods for cleaner construction (M2, M5)` |
| 5 | M3, M10 | `feat: Add comprehensive error context and debug logging (M3, M10)` |
| 6 | M6 | `refactor: Extract animation parameter parsing (M6)` |
| 7 | M9 | `refactor: Reduce cognitive complexity in large methods (M9)` |
| 8 | M11 | `refactor: Encapsulate mutable state in input controllers (M11)` |

---

## Notes

### M1 Closure Rationale
TransformUtil already exists in `menger-common/src/main/scala/menger/common/TransformUtil.scala` with:
- `createScaleTranslation(scale, x, y, z)` 
- `identity()`
- `translation(x, y, z)`
- `uniformScale(scale)`

OptiXEngine already imports and uses `TransformUtil.createScaleTranslation`. No additional work needed.

### M5 Review Findings
Config classes (SceneConfig, OptiXEngineConfig, CameraConfig) already have:
- Clean case class structure with defaults
- Factory methods (`singleObject`, `multiObject`, `Default`, `Testing`)
- No verbose copy chains

M5 work is minimal - just document as complete after verification.

### M11 Risk Assessment
The mutable state in input controllers is required by LibGDX's `InputAdapter` framework. Encapsulation may not provide meaningful improvement since we still need one `var`. Will attempt encapsulation but revert if:
- Code becomes more complex rather than simpler
- Testing doesn't become easier
- Framework constraints prevent clean abstraction
