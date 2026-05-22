# 2. Constraints

## 2.1 Technical Constraints

| Constraint | Description | Background |
|------------|-------------|------------|
| **TC-1: Scala 3** | All application code must use Scala 3 syntax | Modern FP features, no Scala 2 compatibility |
| **TC-2: NVIDIA GPU** | OptiX rendering requires NVIDIA GPU with RTX support | OptiX is NVIDIA-proprietary |
| **TC-3: OptiX SDK 9.0+** | Must match driver version (580.x+ → SDK 9.0) | Version mismatch causes CUDA error 718 |
| **TC-4: CUDA 12.0+** | Required for OptiX compilation | Shader compilation depends on CUDA toolkit |
| **TC-5: No null** | Scala code must use `Option`, `Try`, `Either`. Exception: JNI boundary validation (defensive null checks before native calls) | Enforced by Scalafix; JNI exceptions use `// scalafix:ok` |
| **TC-6: No var/while** | Immutable-only Scala style. See Section 2.5 for documented exceptions | Enforced by Wartremover |
| **TC-7: No throw** | Use `Try` or `Either` for error handling | Enforced by Scalafix |
| **TC-8: 100 char line limit** | Max line length for readability | Enforced by scalafmt |

## 2.2 Organizational Constraints

| Constraint | Description |
|------------|-------------|
| **OC-1: Single developer** | Architecture decisions prioritize simplicity |
| **OC-2: Open source** | Code must be suitable for public repository |
| **OC-3: CI/CD via GitLab** | Pipeline must run on GitLab with GPU runner |

## 2.3 Conventions

| Convention | Description |
|------------|-------------|
| **Naming** | Descriptive function/parameter names over docstrings |
| **Testing** | AnyFlatSpec for all Scala tests |
| **Imports** | One import per line, organized by Scalafix |
| **Alpha** | Standard graphics: 0.0=transparent, 1.0=opaque |

## 2.4 Build Constraints

| Constraint | Description |
|------------|-------------|
| **sbt-jni** | Native code compilation via CMake integration |
| **PTX files** | CUDA shaders compiled to PTX, loaded at runtime |
| **Cross-platform** | Linux x86_64 primary target |

## 2.5 Documented Exceptions to TC-6 (No var)

The following `var` usages are intentional exceptions to the immutability constraint,
each with documented justification. All exceptions use `@SuppressWarnings` annotations.

### 2.5.1 JNI Handle (OptiXRenderer)

| Location | Var | Justification |
|----------|-----|---------------|
| `OptiXRenderer.scala` | `@volatile nativeHandle: Long` | JNI handle pattern — `@volatile` required so the JVM does not cache the value in a register; native code reads/writes this field directly via reflection. No functional alternative exists for JNI interop. |

### 2.5.2 LibGDX Camera Integration (OrbitCamera)

| Location | Var | Justification |
|----------|-----|---------------|
| `menger-app/src/main/scala/menger/input/OrbitCamera.scala` | `eye`, `lookAt`, `up` | LibGDX `Vector3` is inherently mutable by design. Camera state must be updated in-place for LibGDX framework integration. |
| `menger-app/src/main/scala/menger/input/OrbitCamera.scala` | `spherical: SphericalCoords` | Consolidated orbit state (azimuth, elevation, distance). Required for interactive camera control where state changes on every mouse movement. |
| `menger-app/src/main/scala/menger/input/OrbitCamera.scala` | `dragState: Option[CameraDragState]` | Consolidated mouse tracking state. Tracks drag gestures across touchDown/touchDragged/touchUp events. |

### 2.5.3 LibGDX Input State Tracking (KeyPressTracker)

| Location | Var | Justification |
|----------|-----|---------------|
| `menger-app/src/main/scala/menger/input/KeyPressTracker.scala` | `pressed: Map[Key, Boolean]` | Unified modifier key state tracking. LibGDX InputAdapter receives keyDown/keyUp events asynchronously; state must be cached for use in subsequent update() calls. Used by GdxKeyHandler and OptiXKeyHandler for both modifier keys and per-key rotation tracking. |

### Design Rationale

These exceptions share common characteristics:

1. **Framework Integration**: LibGDX is designed around mutable state (Vector3, InputAdapter)
2. **Interactive UI State**: Camera/input controllers must track state across event callbacks
3. **JNI Requirements**: Native interop requires mutable handles accessible to C++ code

The var count has been minimized through consolidation:
- Spherical coords: 3 vars → 1 `SphericalCoords` case class
- Drag state: 4 vars → 1 `Option[DragState]`
