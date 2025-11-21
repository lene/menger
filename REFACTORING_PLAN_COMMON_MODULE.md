# Create `menger-common` Module and Refactor to Vector[3]

## Overview
Create a new `menger-common` SBT subproject to share types between `menger` (root) and `optix-jni`. Move shared types to common module, then refactor coordinate arrays to use `Vector[3]`.

**Status:** ✅ COMPLETE (as of 2025-11-21)

---

## Phase 1: Create `menger-common` Module Structure ✅ COMPLETE

### 1.1 Create Directory Structure ✅
```
menger-common/
├── src/
│   ├── main/scala/menger/common/
│   │   ├── Vector.scala          ✅ moved
│   │   ├── Const.scala           ✅ moved
│   │   ├── ImageSize.scala       ✅ moved
│   │   ├── Light.scala           ✅ unified
│   │   ├── Vec3.scala            ✅ moved
│   │   └── package.scala         ✅ extension methods
│   └── test/scala/menger/common/
│       └── VectorSuite.scala     ✅ moved
```

### 1.2 Update `build.sbt` ✅
Dependencies configured correctly.

---

## Phase 2: Move Core Types to Common Module ✅ COMPLETE

### 2.1 Move `Const.scala` ✅
### 2.2 Move `Vector.scala` ✅
### 2.3 Move `ImageSize` ✅
### 2.4 Move `Vec3` type alias ✅

---

## Phase 3: Unify Light Types ✅ COMPLETE

Unified Light types in `menger-common/src/main/scala/menger/common/Light.scala`

---

## Phase 4: Add Vector[3] Extension Methods ✅ COMPLETE

Extension methods in `menger-common/src/main/scala/menger/common/package.scala`

---

## Phase 5: Refactor OptiXRenderer to Vector[3] ✅ COMPLETE

---

## Phase 6: Update Call Sites ✅ COMPLETE

---

## Phase 7: Cleanup ✅ COMPLETE

---

## Success Criteria

✅ **menger-common module created** with Vector, Const, ImageSize, Light, Vec3
✅ **No circular dependencies** - clean module structure
✅ **Unified Light types** - single implementation, no duplication
✅ **Type-safe coordinates** - Vector[3] with .x, .y, .z accessors
✅ **All tests pass**
