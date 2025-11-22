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
                    │  (LibGDX + OptiX views)     │
                    └─────────────────────────────┘
```

### External Interfaces

| Partner | Interface | Data |
|---------|-----------|------|
| **User** | CLI arguments | `--level`, `--optix`, `--shadows`, etc. |
| **User** | Keyboard/Mouse | Camera control, 4D rotation |
| **File System** | PNG/JPEG files | Rendered frames, animation sequences |
| **GPU** | CUDA/OptiX API | Geometry, shaders, rendered pixels |

## 3.2 Technical Context

```
┌─────────────────────────────────────────────────────────────────┐
│                         Menger Application                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   LibGDX     │    │  Scala Core  │    │   OptiX JNI      │   │
│  │  (OpenGL)    │◄──►│  (Geometry)  │◄──►│  (Ray Tracing)   │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────────┐
│  OpenGL Driver  │                      │   NVIDIA Driver     │
│                 │                      │   + OptiX Runtime   │
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
| **Application** | Scala 3, LibGDX | Core logic, rasterization |
| **Native** | C++17, CUDA, OptiX | GPU ray tracing |
| **Build** | sbt, CMake | Compilation, native integration |
| **Runtime** | JVM 21+, NVIDIA Driver | Execution environment |
