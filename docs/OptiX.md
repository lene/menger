# OptiX Ray Tracing Integration for Menger

## Executive Summary

This document outlines the strategy for integrating NVIDIA OptiX 8.0 ray tracing into the Menger fractal renderer, currently built with Scala 3 and LibGDX. The integration will enable hardware-accelerated ray tracing for superior visual quality and performance, particularly for complex fractal geometries at high iteration levels.

### Key Objectives
- Leverage NVIDIA RTX hardware for accelerated fractal rendering
- Support arbitrarily deep fractal recursion through hybrid rendering techniques
- Maintain compatibility with existing LibGDX renderer as fallback
- Preserve all current rendering modes (wireframe, solid, overlay, fractional levels)
- Enable high-quality offline rendering for animations

### Recommended Approach
After extensive research, the recommended integration strategy is:
1. **Primary**: Custom JNI bridge with C++ OptiX wrapper
2. **Rendering**: Hybrid approach using hardware instancing (levels 0-5) and SDF ray marching (levels 6+)
3. **Fallback**: Maintain LibGDX for non-NVIDIA systems
4. **Future**: Migration path to Panama FFI when mature

## Technical Architecture

### Integration Layer Comparison

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **JCUDA** | Mature Java-CUDA bindings | No OptiX support, only core CUDA | ❌ Not viable |
| **JavaCPP** | Automatic binding generation | No OptiX presets available | ❌ Requires custom work |
| **Panama FFI** | Modern, fast, part of JDK 22+ | Still experimental for complex APIs | 🔄 Future option |
| **Custom JNI** | Full control, proven approach | More development effort | ✅ **Recommended** |

### Rendering Strategy

#### Hybrid Rendering Modes

**Mode A: Hardware-Accelerated Instancing (Levels 0-5)**
- Use OptiX Instance Acceleration Structure (IAS)
- Single Geometry Acceleration Structure (GAS) for base unit
- Transform matrices for each fractal element
- Full BVH hardware traversal
- Memory: ~48 bytes per instance

**Mode B: SDF Ray Marching (Levels 6+)**
- Custom intersection program with distance fields
- No memory growth with iteration level
- Supports infinite detail zoom
- Natural fractional level support
- Performance depends on march step count

### System Architecture

```
┌─────────────────────────────────────────────────┐
│              Scala Application Layer            │
├─────────────────────────────────────────────────┤
│                 MengerEngine                     │
│  ┌──────────────┐          ┌─────────────────┐ │
│  │ LibGDXEngine │          │  OptiXEngine    │ │
│  └──────────────┘          └─────────────────┘ │
├─────────────────────────────────────────────────┤
│              Geometry Generation                 │
│  (SpongeBySurface, TesseractSponge, etc.)      │
├─────────────────────────────────────────────────┤
│                 JNI Bridge                       │
│  ┌────────────────────────────────────────────┐ │
│  │ OptiXRenderer (Scala) → liboptixjni.so     │ │
│  └────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────┤
│              C++/CUDA Layer                      │
│  ┌────────────────────────────────────────────┐ │
│  │ OptiXWrapper: Context, Pipeline, Launch    │ │
│  │ Shaders: RayGen, Hit, Miss, Intersection   │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## OptiX Pipeline Design

### Core Components

#### 1. OptiX Context and Pipeline Setup
```cpp
class OptiXWrapper {
private:
    OptixDeviceContext context;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    OptixModule module;

    // Acceleration structures
    OptixTraversableHandle gas_handle;  // Base geometry
    OptixTraversableHandle ias_handle;  // Instances

    // Device memory
    CUdeviceptr d_instances;
    CUdeviceptr d_vertices;
    CUdeviceptr d_output;

public:
    void initialize();
    void buildGeometry(float* vertices, int* indices, int numFaces);
    void buildInstances(float* transforms, int numInstances);
    void render(Camera camera, uint32_t* output);
};
```

#### 2. Shader Programs

**Ray Generation (menger_raygen.cu)**
```cuda
extern "C" __global__ void __raygen__camera() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Generate camera ray
    float2 d = 2.0f * make_float2(
        float(idx.x) / float(dim.x),
        float(idx.y) / float(dim.y)
    ) - 1.0f;

    float3 ray_origin = params.cam_eye;
    float3 ray_direction = normalize(
        d.x * params.cam_u +
        d.y * params.cam_v +
        params.cam_w
    );

    // Trace ray
    RayPayload payload;
    optixTrace(params.handle, ray_origin, ray_direction,
               0.01f, 1e16f, 0.0f,
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0, 1, 0,
               payload);

    // Write output
    params.image[idx.y * params.width + idx.x] =
        make_color(payload.color);
}
```

**Closest Hit (menger_closesthit.cu)**
```cuda
extern "C" __global__ void __closesthit__solid() {
    HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

    // Get triangle normal
    float3 normal = getTriangleNormal();

    // Phong shading
    float3 light_dir = normalize(params.light_direction);
    float ndotl = max(0.0f, dot(normal, light_dir));
    float3 diffuse = data->albedo * (0.2f + 0.8f * ndotl);

    // Set payload
    setPayloadColor(diffuse);
}
```

**SDF Intersection (menger_sdf.cu)**
```cuda
__device__ float mengerSpongeSDF(float3 p, int iterations) {
    float scale = 1.0f;
    for (int i = 0; i < iterations; i++) {
        p = fmodf(p * 3.0f + 1.0f, 2.0f) - 1.0f;
        scale *= 3.0f;

        float da = max(fabs(p.x), fabs(p.y));
        float db = max(fabs(p.y), fabs(p.z));
        float dc = max(fabs(p.z), fabs(p.x));
        float d = min(da, min(db, dc)) - 0.25f;

        if (d < 0.0f) return d / scale;
    }
    return (length(p) - 0.5f) / scale;
}

extern "C" __global__ void __intersection__menger_sdf() {
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float tmin = optixGetRayTmin();
    float tmax = optixGetRayTmax();

    // Ray marching
    float t = tmin;
    for (int steps = 0; steps < 256 && t < tmax; steps++) {
        float3 pos = ray_orig + t * ray_dir;
        float dist = mengerSpongeSDF(pos, params.sdf_iterations);

        if (dist < 0.001f) {
            // Compute normal via gradient
            float3 normal = computeNormalGradient(pos);
            optixReportIntersection(t, 0,
                float3_as_uints(normal));
            return;
        }
        t += dist * 0.9f;  // Conservative step
    }
}
```

### Memory Management

#### Instance Representation (16 bytes)
```cpp
struct CompactInstance {
    float3 position;      // 12 bytes
    uint8_t scale_exp;    // 1 byte (scale = 3^(-scale_exp))
    uint8_t material_id;  // 1 byte
    uint16_t flags;       // 2 bytes
};
```

#### Memory Requirements by Level
| Level | 3D Instances | 4D Instances | Memory (16-byte) | Memory (48-byte) |
|-------|-------------|--------------|------------------|-------------------|
| 0 | 6 | 8 | 96 B | 288 B |
| 1 | 72 | 384 | 1.1 KB | 3.4 KB |
| 2 | 864 | 18,432 | 13.5 KB | 40.5 KB |
| 3 | 10,368 | 884,736 | 162 KB | 486 KB |
| 4 | 124,416 | 42M | 1.9 MB | 5.8 MB |
| 5 | 1.5M | 2B | 23 MB | 70 MB |
| 6 | 18M | - | 280 MB | 840 MB |

### Fractal-Specific Features

#### Fractional Level Rendering
```cuda
// Render floor and ceiling levels with alpha blending
struct FractionalLevel {
    float level;
    float floor_alpha;   // 1.0 - fract(level)
    float ceil_alpha;    // 1.0
};

// In any-hit shader for transparency
extern "C" __global__ void __anyhit__transparent() {
    float alpha = getInstanceAlpha();
    float xi = random(getPayloadSeed());
    if (xi > alpha) {
        optixIgnoreIntersection();
    }
}
```

#### 4D Tesseract Support
```scala
// CPU-side projection before OptiX
class OptiXTesseractRenderer {
  def projectAndRender(mesh4D: Mesh4D, params: RotationProjectionParameters): Unit = {
    // Apply 4D rotations
    val rotated = mesh4D.rotate4D(params.rotation)

    // Project to 3D
    val projected3D = rotated.projectTo3D(params.projection)

    // Convert to OptiX format
    val instances = projected3D.toInstanceTransforms()

    // Send to OptiX
    optiXRenderer.setInstances(instances)
    optiXRenderer.render()
  }
}
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Objective**: Establish OptiX infrastructure and basic rendering

**Tasks**:
- [ ] Set up CMake build system for C++/CUDA
- [ ] Create JNI interface definitions in Scala
- [ ] Implement OptiXWrapper initialization
- [ ] Compile basic shader programs (raygen, miss, closesthit)
- [ ] Test with simple cube geometry
- [ ] Integrate with sbt build via sbt-jni plugin

**Deliverables**:
- Working OptiX context creation
- Single cube ray traced rendering
- JNI bridge functional

### Phase 2: Fractal Instancing (Week 3-4)
**Objective**: Implement fractal rendering with hardware acceleration

**Tasks**:
- [ ] Generate instance transforms for Menger sponge
- [ ] Build two-level acceleration structure (GAS + IAS)
- [ ] Optimize instance memory layout
- [ ] Implement material system in SBT
- [ ] Add wireframe rendering support
- [ ] Performance profiling and optimization

**Deliverables**:
- Levels 0-4 rendering with instancing
- Performance benchmarks
- Memory usage analysis

### Phase 3: Advanced Features (Week 5-6)
**Objective**: Achieve feature parity with LibGDX renderer

**Tasks**:
- [ ] Implement fractional level support
- [ ] Add transparency/alpha blending
- [ ] Integrate 4D tesseract rendering
- [ ] Implement SDF ray marching for high levels
- [ ] Add overlay mode (solid + wireframe)
- [ ] Interactive camera controls

**Deliverables**:
- All sponge types supported
- Fractional levels working
- SDF mode for levels 6+

### Phase 4: Integration & Polish (Week 7)
**Objective**: Production-ready integration

**Tasks**:
- [ ] LibGDX texture sharing
- [ ] Animation rendering pipeline
- [ ] Comprehensive testing
- [ ] Documentation
- [ ] Performance tuning
- [ ] CI/CD integration

**Deliverables**:
- Feature-complete OptiX renderer
- User documentation
- Performance report

## Build System Integration

### SBT Configuration
```scala
// build.sbt additions
lazy val optixJni = (project in file("optix-jni"))
  .enablePlugins(JniNative)
  .settings(
    nativePlatforms := Set("linux-x86_64"),
    nativeCompile / sourceDirectory := sourceDirectory.value / "native",
    nativeCompile / target := target.value / "native" / nativePlatform.value,
    jniLibraryName := "optixjni"
  )

lazy val root = (project in file("."))
  .dependsOn(optixJni)
  .settings(
    javaOptions ++= Seq(
      s"-Djava.library.path=${(optixJni / nativeCompile / target).value}"
    )
  )
```

### CMake Configuration
```cmake
# optix-jni/CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(optixjni LANGUAGES CXX CUDA)

find_package(OptiX REQUIRED)
find_package(CUDA REQUIRED)
find_package(JNI REQUIRED)

# Compile CUDA kernels to PTX
add_library(shaders OBJECT
  shaders/menger_raygen.cu
  shaders/menger_closesthit.cu
  shaders/menger_sdf.cu
)
set_property(TARGET shaders PROPERTY CUDA_PTX_COMPILATION ON)

# Build JNI library
add_library(optixjni SHARED
  src/OptiXWrapper.cpp
  src/JNIBindings.cpp
)
target_include_directories(optixjni PRIVATE
  ${OptiX_INCLUDE}
  ${JNI_INCLUDE_DIRS}
)
target_link_libraries(optixjni
  ${CUDA_LIBRARIES}
  ${OptiX_LIBRARIES}
)
```

## Performance Optimization Strategies

### Level-of-Detail (LOD)
```cpp
int computeLOD(float distance_to_camera, int base_level) {
    // Reduce detail for distant objects
    float detail_factor = 100.0f;
    int lod_reduction = log2f(distance_to_camera / detail_factor);
    return max(0, base_level - lod_reduction);
}
```

### Frustum Culling
```scala
def cullInstances(instances: Array[Instance], frustum: Frustum): Array[Instance] = {
  instances.filter { inst =>
    val bounds = computeBounds(inst.transform, baseGeometry.bounds)
    frustum.intersects(bounds)
  }
}
```

### Streaming for High Levels
```cpp
// Generate instances in batches to avoid memory spikes
class StreamingInstanceBuilder {
    void buildLevel(int level) {
        const int BATCH_SIZE = 100000;
        int total = pow(20, level);

        for (int offset = 0; offset < total; offset += BATCH_SIZE) {
            int count = min(BATCH_SIZE, total - offset);
            generateBatch(offset, count);
            uploadToDevice();
            updateIAS();
        }
    }
};
```

## Testing Strategy

### Unit Tests
- JNI bridge functionality
- Geometry generation correctness
- Transform matrix calculations
- Memory management

### Integration Tests
```bash
# Test all sponge types
xvfb-run -a ./test-optix --sponge-type cube --level 3
xvfb-run -a ./test-optix --sponge-type tesseract-sponge --level 2
xvfb-run -a ./test-optix --sponge-type tesseract-sponge-2 --level 1.5

# Performance benchmarks
./benchmark-optix --compare-with-libgdx
```

### Performance Targets
| Scenario | Resolution | Target FPS | LibGDX FPS |
|----------|------------|------------|------------|
| Level 3 Menger | 1920x1080 | 60+ | 30-45 |
| Level 4 Menger | 1920x1080 | 30-60 | 10-20 |
| Level 5 Menger | 1920x1080 | 10-30 | 2-5 |
| Level 6 (SDF) | 1920x1080 | 5-15 | N/A |

## Risk Assessment & Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| JNI performance overhead | Medium | Low | Direct ByteBuffer usage, minimize transfers |
| Memory limits for high levels | High | Medium | SDF fallback, streaming, LOD |
| Build complexity | Low | Medium | Docker images, CI automation |
| Platform lock-in | Medium | Low | LibGDX fallback, abstraction layer |
| OptiX API changes | Low | Low | Version pinning, wrapper abstraction |

### Fallback Strategies
1. **No NVIDIA GPU**: Automatically use LibGDX renderer
2. **Insufficient VRAM**: Switch to SDF mode or reduce level
3. **OptiX initialization failure**: Graceful degradation with warning
4. **Performance issues**: Adaptive quality settings

## Future Enhancements

### Short-term (3-6 months)
- [ ] Multi-GPU support for large scenes
- [ ] Denoising for path-traced mode
- [ ] Motion blur for animations
- [ ] Depth of field effects

### Long-term (6-12 months)
- [ ] Panama FFI migration when stable
- [ ] WebGPU export for browser rendering
- [ ] Volumetric rendering for fractals
- [ ] Real-time collaborative viewing

### Research Opportunities
- Procedural texture generation on GPU
- AI-assisted LOD selection
- Compression algorithms for instance data
- Novel fractal types optimized for ray tracing

## API Documentation

### Scala Interface
```scala
package menger.optix

trait OptiXRenderer {
  /**
   * Initialize OptiX context and pipeline
   * @return Success or error with details
   */
  def initialize(): Try[Unit]

  /**
   * Set fractal geometry instances
   * @param instances Array of 4x3 transform matrices
   */
  def setInstances(instances: Array[Matrix4x3]): Unit

  /**
   * Render frame with given camera
   * @param camera Camera parameters
   * @return RGBA image data
   */
  def render(camera: Camera): Array[Byte]

  /**
   * Update rendering parameters
   * @param params Rendering settings
   */
  def updateParams(params: RenderParams): Unit

  /**
   * Clean up OptiX resources
   */
  def dispose(): Unit
}

case class RenderParams(
  maxDepth: Int = 2,
  samplesPerPixel: Int = 1,
  enableDenoising: Boolean = false,
  renderMode: RenderMode = RenderMode.Solid
)

sealed trait RenderMode
object RenderMode {
  case object Solid extends RenderMode
  case object Wireframe extends RenderMode
  case object Overlay extends RenderMode
}
```

### C++ Interface
```cpp
// optix-jni/include/OptiXWrapper.h
class OptiXWrapper {
public:
    OptiXWrapper();
    ~OptiXWrapper();

    bool initialize();
    void setInstances(const float* transforms, size_t count);
    void render(const Camera& camera, uint32_t* output);
    void updateParams(const RenderParams& params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
```

## Conclusion

This document provides a comprehensive plan for integrating NVIDIA OptiX ray tracing into the Menger fractal renderer. The hybrid approach of hardware-accelerated instancing combined with SDF ray marching offers the best balance of performance, quality, and flexibility. The phased implementation plan allows for incremental development and testing while maintaining the existing LibGDX renderer as a fallback option.

The integration will enable:
- Superior rendering quality with proper lighting and shadows
- Support for arbitrarily deep fractal levels
- High-performance animation generation
- Future-proof architecture adaptable to new GPU technologies

Next steps:
1. Review and approve this plan
2. Set up development environment with OptiX SDK
3. Begin Phase 1 implementation
4. Iterate based on performance testing and user feedback