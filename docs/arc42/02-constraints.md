# 2. Constraints

## 2.1 Technical Constraints

| Constraint | Description | Background |
|------------|-------------|------------|
| **TC-1: Scala 3** | All application code must use Scala 3 syntax | Modern FP features, no Scala 2 compatibility |
| **TC-2: NVIDIA GPU** | OptiX rendering requires NVIDIA GPU with RTX support | OptiX is NVIDIA-proprietary |
| **TC-3: OptiX SDK 9.0+** | Must match driver version (580.x+ → SDK 9.0) | Version mismatch causes CUDA error 718 |
| **TC-4: CUDA 12.0+** | Required for OptiX compilation | Shader compilation depends on CUDA toolkit |
| **TC-5: No null** | Scala code must use `Option`, `Try`, `Either` | Enforced by Wartremover and Scalafix |
| **TC-6: No var/while** | Immutable-only Scala style | Enforced by Wartremover |
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
