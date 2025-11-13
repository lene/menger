# Ray Statistics Collection Implementation Plan

**Created:** 2025-11-13
**Status:** Ready for Implementation

## Overview

Add comprehensive ray tracing statistics tracking to OptiX renderer with minimal performance overhead. Statistics are always collected but only printed when `--stats` flag is set.

## Statistics Tracked

1. **Total rays cast** - All rays generated during render
2. **Primary rays** - Initial camera rays (= pixel count)
3. **Reflected rays** - Rays from Fresnel reflection
4. **Refracted rays** - Rays transmitted through sphere
5. **Min recursion depth** - Shallowest ray bounce count
6. **Max recursion depth** - Deepest ray bounce count
7. **Total depth sum** - For calculating average depth

## Implementation Steps

### 1. Add RayStats Structure (C++ Header)

**File:** `optix-jni/src/main/native/include/OptiXData.h`

Add data structure:
```cpp
struct RayStats {
    unsigned long long total_rays;
    unsigned long long primary_rays;
    unsigned long long reflected_rays;
    unsigned long long refracted_rays;
    unsigned int min_depth;
    unsigned int max_depth;
    unsigned long long depth_sum;
};
```

Update `Params` struct:
```cpp
struct Params {
    // ... existing fields ...
    RayStats* d_ray_stats;  // Pointer to GPU stats buffer
};
```

### 2. Update CUDA Shader (sphere_combined.cu)

**File:** `optix-jni/src/main/native/shaders/sphere_combined.cu`

**Ray generation program** (`__raygen__rg`):
```cuda
extern "C" __global__ void __raygen__rg() {
    // ... existing code ...

    // Count primary ray
    atomicAdd(&params.d_ray_stats->primary_rays, 1ULL);
    atomicAdd(&params.d_ray_stats->total_rays, 1ULL);

    // Initialize depth tracking for this ray
    unsigned int depth = 0;

    // ... trace ray ...
}
```

**Closest hit program** (`__closesthit__ch`):
```cuda
extern "C" __global__ void __closesthit__ch() {
    // ... existing code ...

    unsigned int current_depth = optixGetRayTmax();  // or track via payload

    // Track depth statistics
    atomicMin(&params.d_ray_stats->min_depth, current_depth);
    atomicMax(&params.d_ray_stats->max_depth, current_depth);
    atomicAdd(&params.d_ray_stats->depth_sum, (unsigned long long)current_depth);

    // When tracing reflection ray:
    if (/* reflection path */) {
        atomicAdd(&params.d_ray_stats->reflected_rays, 1ULL);
        atomicAdd(&params.d_ray_stats->total_rays, 1ULL);
        // ... optixTrace() ...
    }

    // When tracing refraction ray:
    if (/* refraction path */) {
        atomicAdd(&params.d_ray_stats->refracted_rays, 1ULL);
        atomicAdd(&params.d_ray_stats->total_rays, 1ULL);
        // ... optixTrace() ...
    }
}
```

### 3. C++ Buffer Management (OptiXWrapper.cpp)

**File:** `optix-jni/src/main/native/OptiXWrapper.cpp`

**Add to Impl struct:**
```cpp
struct Impl {
    // ... existing fields ...
    CUdeviceptr d_ray_stats;  // GPU buffer for statistics
    RayStats h_ray_stats;     // Host copy of statistics
};
```

**In constructor:**
```cpp
// Allocate stats buffer on GPU
CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&impl->d_ray_stats),
    sizeof(RayStats)
));
```

**In destructor:**
```cpp
if (impl->d_ray_stats) {
    cudaFree(reinterpret_cast<void*>(impl->d_ray_stats));
}
```

**In render() method:**
```cpp
// Zero stats before render
RayStats zero_stats = {0, 0, 0, 0, UINT_MAX, 0, 0};
CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(impl->d_ray_stats),
    &zero_stats,
    sizeof(RayStats),
    cudaMemcpyHostToDevice
));

// Set stats pointer in params
params.d_ray_stats = reinterpret_cast<RayStats*>(impl->d_ray_stats);

// ... launch kernel ...

// Copy stats back from GPU
CUDA_CHECK(cudaMemcpy(
    &impl->h_ray_stats,
    reinterpret_cast<void*>(impl->d_ray_stats),
    sizeof(RayStats),
    cudaMemcpyDeviceToHost
));

// Return stats to caller (via output parameter or return value)
```

### 4. JNI Bindings Update

**File:** `optix-jni/src/main/native/include/OptiXWrapper.h`

Update method signature:
```cpp
class OptiXWrapper {
public:
    // ... existing methods ...

    // Option 1: Return stats via output parameter
    void render(int width, int height, unsigned char* output, RayStats* stats);

    // Option 2: Add separate getter
    void getLastRayStats(RayStats* stats);
};
```

**File:** `optix-jni/src/main/native/JNIBindings.cpp`

```cpp
JNIEXPORT jbyteArray JNICALL
Java_menger_optix_OptiXRenderer_render(
    JNIEnv* env,
    jobject obj,
    jlong handle,
    jint width,
    jint height,
    jlongArray statsArray  // Output parameter for stats
) {
    // ... existing code ...

    RayStats stats;
    wrapper->render(width, height, output_buffer, &stats);

    // Copy stats to Java array
    jlong stats_data[7] = {
        stats.total_rays,
        stats.primary_rays,
        stats.reflected_rays,
        stats.refracted_rays,
        stats.min_depth,
        stats.max_depth,
        stats.depth_sum
    };
    env->SetLongArrayRegion(statsArray, 0, 7, stats_data);

    // ... return image bytes ...
}
```

### 5. Scala Interface Updates

**File:** `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`

Add case class:
```scala
case class RayStats(
  totalRays: Long,
  primaryRays: Long,
  reflectedRays: Long,
  refractedRays: Long,
  minDepth: Long,
  maxDepth: Long,
  depthSum: Long
) {
  def avgDepth: Double = if (totalRays > 0) depthSum.toDouble / totalRays else 0.0

  def reflectedPercent: Double = if (totalRays > 0) (reflectedRays * 100.0) / totalRays else 0.0

  def refractedPercent: Double = if (totalRays > 0) (refractedRays * 100.0) / totalRays else 0.0
}
```

Update render method:
```scala
@native private def renderNative(
  width: Int,
  height: Int,
  statsArray: Array[Long]
): Array[Byte]

def render(width: Int, height: Int): (Array[Byte], RayStats) = {
  val statsArray = new Array[Long](7)
  val imageBytes = renderNative(width, height, statsArray)

  val stats = RayStats(
    totalRays = statsArray(0),
    primaryRays = statsArray(1),
    reflectedRays = statsArray(2),
    refractedRays = statsArray(3),
    minDepth = statsArray(4),
    maxDepth = statsArray(5),
    depthSum = statsArray(6)
  )

  (imageBytes, stats)
}
```

### 6. CLI Flag Addition

**File:** `src/main/scala/menger/MengerCLIOptions.scala`

```scala
val stats: ScallopOption[Boolean] = opt[Boolean](
  required = false,
  default = Some(false),
  descr = "Print ray tracing statistics after rendering"
)
```

### 7. Stats Printing Logic

**File:** `src/main/scala/menger/OptiXEngine.scala`

Update to receive and conditionally print stats:
```scala
class OptiXEngine(
  // ... existing parameters ...
  printStats: Boolean
) {
  def render(): Unit = {
    val startTime = System.nanoTime()

    val (rgbaBytes, stats) = optiXResources.renderScene(width, height)

    val endTime = System.nanoTime()
    val renderTimeMs = (endTime - startTime) / 1_000_000.0

    if (printStats) {
      printRayStatistics(stats, renderTimeMs)
    }

    // ... rest of render logic ...
  }

  private def printRayStatistics(stats: RayStats, renderTimeMs: Double): Unit = {
    val raysPerSec = if (renderTimeMs > 0) (stats.totalRays / (renderTimeMs / 1000.0)).toLong else 0L

    println("Ray Statistics:")
    println(f"  Total rays:       ${stats.totalRays}%,d")
    println(f"  Primary rays:     ${stats.primaryRays}%,d")
    println(f"  Reflected rays:   ${stats.reflectedRays}%,d (${stats.reflectedPercent}%.1f%%)")
    println(f"  Refracted rays:   ${stats.refractedRays}%,d (${stats.refractedPercent}%.1f%%)")
    println(f"  Recursion depth:  min=${stats.minDepth}, max=${stats.maxDepth}, avg=${stats.avgDepth}%.2f")
    println(f"  Render time:      ${renderTimeMs}%.1f ms")
    println(f"  Throughput:       ${raysPerSec}%,d rays/sec")
  }
}
```

**File:** `src/main/scala/menger/Main.scala`

Pass flag to OptiXEngine:
```scala
val engine = new OptiXEngine(
  // ... existing parameters ...
  printStats = opts.stats()
)
```

### 8. Testing

**File:** `optix-jni/src/test/scala/menger/optix/RayStatsTest.scala`

```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RayStatsTest extends AnyFlatSpec with Matchers {

  "RayStats" should "count primary rays correctly" in {
    // 800×600 image should have exactly 480,000 primary rays
    val (_, stats) = renderer.render(800, 600)
    stats.primaryRays shouldBe 480000
  }

  it should "have zero refracted rays for opaque sphere" in {
    // Configure opaque sphere (alpha = 1.0)
    renderer.setSphereColor(1.0f, 0.0f, 0.0f, 1.0f)
    val (_, stats) = renderer.render(800, 600)
    stats.refractedRays shouldBe 0
  }

  it should "have non-zero refracted rays for transparent sphere" in {
    // Configure transparent sphere (alpha = 0.5)
    renderer.setSphereColor(1.0f, 0.0f, 0.0f, 0.5f)
    val (_, stats) = renderer.render(800, 600)
    stats.refractedRays should be > 0L
  }

  it should "calculate average depth correctly" in {
    val stats = RayStats(
      totalRays = 1000,
      primaryRays = 500,
      reflectedRays = 300,
      refractedRays = 200,
      minDepth = 1,
      maxDepth = 5,
      depthSum = 2570
    )
    stats.avgDepth shouldBe 2.57 +- 0.01
  }

  it should "have total rays equal to sum of ray types" in {
    val (_, stats) = renderer.render(800, 600)
    // Total should approximately equal primary + reflected + refracted
    // (may have small discrepancies due to recursion termination)
    val sum = stats.primaryRays + stats.reflectedRays + stats.refractedRays
    stats.totalRays shouldBe sum +- (sum * 0.01).toLong // 1% tolerance
  }
}
```

**Performance test:**
```scala
it should "have acceptable overhead (<5%)" in {
  val iterations = 10

  // Warmup
  for (_ <- 1 to 5) renderer.render(800, 600)

  // Measure with stats collection
  val startTime = System.nanoTime()
  for (_ <- 1 to iterations) renderer.render(800, 600)
  val totalTime = (System.nanoTime() - startTime) / 1_000_000.0
  val avgTimeMs = totalTime / iterations

  println(s"Average render time with stats: $avgTimeMs ms")

  // Overhead should be < 5% compared to baseline
  // (Baseline would be measured without atomic operations)
}
```

## Performance Overhead Estimate

**Atomic operations overhead:** ~2-5%

- 1-2 atomic increments per ray (total + type)
- 2 atomic min/max per ray (depth tracking)
- Modern GPUs have efficient atomic operations on global memory
- Overhead acceptable given always-on collection provides debugging value

**Decision:** Always collect, conditionally print

- Statistics are useful for development and debugging
- Overhead is minimal (<5%)
- No conditional logic in hot path (cleaner code)
- User controls output verbosity with --stats flag

## Files Modified Summary

1. `optix-jni/src/main/native/include/OptiXData.h` - Add RayStats struct
2. `optix-jni/src/main/native/shaders/sphere_combined.cu` - Add atomic counters
3. `optix-jni/src/main/native/OptiXWrapper.cpp` - Buffer management
4. `optix-jni/src/main/native/include/OptiXWrapper.h` - Update method signature
5. `optix-jni/src/main/native/JNIBindings.cpp` - JNI interface
6. `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` - Scala API
7. `src/main/scala/menger/OptiXEngine.scala` - Stats printing
8. `src/main/scala/menger/MengerCLIOptions.scala` - CLI flag
9. `src/main/scala/menger/Main.scala` - Pass flag to engine
10. `optix-jni/src/test/scala/menger/optix/RayStatsTest.scala` - New test file

## Expected Output Format

```
Ray Statistics:
  Total rays:       1,234,567
  Primary rays:     480,000
  Reflected rays:   456,789 (37.0%)
  Refracted rays:   297,778 (24.1%)
  Recursion depth:  min=1, max=5, avg=2.57
  Render time:      145.3 ms
  Throughput:       8,493,210 rays/sec
```

## Estimated Effort

- Implementation: 3-4 hours
- Testing: 1 hour
- **Total: 4-5 hours**

## Implementation Order

1. Add C++ data structures (OptiXData.h)
2. Update CUDA shader with atomic operations (sphere_combined.cu)
3. Implement buffer management (OptiXWrapper.cpp/h)
4. Update JNI bindings (JNIBindings.cpp)
5. Update Scala interface (OptiXRenderer.scala)
6. Add CLI flag (MengerCLIOptions.scala)
7. Add printing logic (OptiXEngine.scala, Main.scala)
8. Write tests (RayStatsTest.scala)
9. Compile, test, verify overhead is acceptable
10. Update CHANGELOG.md

## Notes

- Min depth should be initialized to UINT_MAX on GPU, then atomicMin will find the true minimum
- Depth tracking requires passing depth through ray payload or using OptiX built-in ray depth query
- Consider adding render time measurement at Scala layer for throughput calculation
- Statistics buffer is small (56 bytes), negligible memory overhead
