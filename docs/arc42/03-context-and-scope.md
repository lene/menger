# 3. Context and Scope

## 3.1 Business Context

```
                    ┌─────────────────────────────────────┐
                    │              Menger                 │
                    │         Fractal Renderer            │
                    └─────────────────────────────────────┘
                              ▲           │
                              │           │
                    CLI args  │           │ Images/Frames
                              │           ▼
┌──────────┐         ┌────────┴───────────────────┐
│   User   │◄───────►│     Command Line (sbt run) │
└──────────┘         └────────────────────────────┘
                              │
                              │ Interactive
                              ▼
                    ┌─────────────────────────────┐
                    │    Interactive Window       │
                    │  (LibGDX/LWJGL3 + OptiX)   │
                    └─────────────────────────────┘
```

### External Interfaces

| Partner | Interface | Data |
|---------|-----------|------|
| **User** | CLI arguments | `--level`, `--shadows`, `--objects`, `--scene`, etc. |
| **User** | Keyboard/Mouse | Camera control, 4D rotation |
| **File System** | PNG/JPEG files | Rendered frames, animation sequences |
| **GPU** | CUDA/OptiX API | Geometry, shaders, rendered pixels |

## 3.2 Technical Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         Menger Application                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   LibGDX /   │    │  Scala Core  │    │   OptiX JNI      │   │
│  │  LWJGL3      │◄──►│  (Geometry)  │◄──►│  (Ray Tracing)   │   │
│  │  (windowing) │    │              │    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────────┐
│  OpenGL/LWJGL3  │                      │   NVIDIA Driver     │
│  (window only)  │                      │   + OptiX Runtime   │
└─────────────────┘                      └─────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         NVIDIA GPU Hardware                      │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Application** | Scala 3, LibGDX/LWJGL3 | Core logic, windowing, input |
| **Native** | C++17, CUDA, OptiX | GPU ray tracing |
| **Build** | sbt, CMake | Compilation, native integration |
| **Runtime** | JVM 21+, NVIDIA Driver | Execution environment |

### External Services

Boundaries where Menger meets non-Menger code. See
`docs/ARCHITECTURE_MODULES.md` for the single-page snapshot; this table is
the canonical arc42 record.

| Service | Crossed by | Direction | Notes |
|---------|------------|-----------|-------|
| NVIDIA OptiX (8.x) + CUDA driver | `optix-jni` native C++ (`OptiXContext`, `OptiXWrapper`) | sync calls | GPU device context, BVH builds, ray launches, photon-mapping passes. |
| LibGDX 1.13 + LWJGL3 | `Main.scala` (window config), `menger.input` (input adapter) | sync calls + callbacks | Windowing, OpenGL context, keyboard/mouse events. Mutable LibGDX state is confined to `menger.input`. |
| FFmpeg (external binary) | `menger.engines.VideoEncoder` | spawned subprocess | Used by `VideoEngine` to encode rendered frames. Availability is checked at engine construction; absence throws. |
| File system | `menger.TextureLoader`, `menger.dsl.SceneLoader`, `menger.engines.SavesScreenshots`, `ScreenshotFactory` | read / write | Texture files, scene `.scala` files, PNG screenshots, video output. |
| SLF4J / Logback | application-wide | sync log calls | Configured at `Main` start. Banned in `menger.common`; partly violated in `menger.objects` (M-arch-objects-logging). |
| Scallop | `MengerCLIOptions`, `menger.cli` | construction-time | CLI argument parsing. |
| ArchUnit | `menger-app` test sources only | test-only | Enforces the onion layering documented in §5 and AD-23. |
