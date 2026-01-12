# Sprint 12: Object Animation Foundation

**Sprint:** 12 - Object Animation Foundation
**Status:** Not Started
**Estimate:** 12-18 hours
**Branch:** `feature/sprint-12`
**Dependencies:** Sprint 11 (Scala DSL) - recommended but not blocking

---

## Goal

Enable animated scene rendering in OptiX mode with frame-by-frame object transformation, outputting to image sequences for video creation.

## Success Criteria

- [ ] Animate object positions over time with keyframes
- [ ] Animate object rotations over time with keyframes
- [ ] Animate object scale over time with keyframes
- [ ] Linear interpolation between keyframes
- [ ] Output image sequence (PNG) with frame numbering
- [ ] CLI: `--animate-scene --frames N --output pattern%04d.png`
- [ ] DSL integration: `sphere(Glass).animate(...)` syntax (if Sprint 11 complete)
- [ ] All tests pass (~30 new tests)

---

## Scope

### In Scope (First Version)
- Position animation (translate objects over time)
- Rotation animation (Euler angles, linear interpolation)
- Scale animation (uniform scaling)
- Linear interpolation between keyframes
- Frame sequence output (PNG)
- CLI options for animation control
- Basic DSL animation syntax

### Deferred to Sprint 13
- Easing functions (ease-in-out, cubic, bounce)
- Camera animation (path following)
- Light animation
- Quaternion SLERP for rotation (smoother rotation)
- Animation preview mode
- Video output (MP4/WebM)

---

## Background

### Existing Animation Infrastructure

The codebase has animation support for LibGDX mode only:

| Class | Location | Purpose |
|-------|----------|---------|
| `AnimationSpecification` | `menger/AnimationSpecification.scala` | Parses `frames=N:param=start-end` format |
| `AnimationSpecificationSequence` | `menger/AnimationSpecificationSequence.scala` | Chains animation segments |
| `AnimatedMengerEngine` | `menger/engines/AnimatedMengerEngine.scala` | LibGDX animation render loop |
| `SavesScreenshots` | `menger/engines/SavesScreenshots.scala` | Screenshot saving trait |

**Gap:** OptiX mode has no animation support. The existing animation only affects 4D rotation/projection parameters, not object transforms.

### OptiX Instance API

Current instance management in `OptiXRenderer`:
```scala
def addSphereInstance(transform: Array[Float], color: Color, ior: Float): Option[Int]
def addTriangleMeshInstance(transform: Array[Float], color: Color, ior: Float, textureIndex: Int): Option[Int]
def removeInstance(instanceId: Int): Unit
def clearAllInstances(): Unit
def getInstanceCount(): Int
```

**Missing:** `updateInstanceTransform(instanceId, transform)` - required to modify transforms per frame without recreating instances.

### Transform Matrix Format

OptiX uses 4x3 row-major matrices (12 floats):
```
[sx,  0,  0, tx]   // row 0: scale-x, translate-x
[ 0, sy,  0, ty]   // row 1: scale-y, translate-y
[ 0,  0, sz, tz]   // row 2: scale-z, translate-z
```

For rotation, the matrix becomes:
```
[r00, r01, r02, tx]
[r10, r11, r12, ty]
[r20, r21, r22, tz]
```

---

## Tasks

### Step 12.1: Add Transform Update JNI Method

**Status:** Not Started
**Estimate:** 2.5 hours

Add native method to update instance transforms without recreating instances.

#### Files to Modify

**`optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala`**

Add after line 353 (after `getInstanceCount`):

```scala
/**
 * Update the transform matrix for an existing instance.
 * This is more efficient than remove+add for animation.
 *
 * @param instanceId ID returned from addXxxInstance
 * @param transform 12-element array (4x3 row-major matrix)
 * @return true if update succeeded, false if instanceId invalid
 */
@native private def updateInstanceTransformNative(instanceId: Int, transform: Array[Float]): Boolean

def updateInstanceTransform(instanceId: Int, transform: Array[Float]): Boolean =
  require(
    transform.length == Const.Renderer.transformMatrixSize,
    s"Transform must have ${Const.Renderer.transformMatrixSize} elements (4x3 matrix), got ${transform.length}"
  )
  updateInstanceTransformNative(instanceId, transform)

/**
 * Rebuild the Instance Acceleration Structure after transform updates.
 * Must be called after updating transforms and before rendering.
 */
@native def rebuildIAS(): Unit

/**
 * Batch update multiple instance transforms efficiently.
 * Rebuilds IAS once after all updates.
 *
 * @param updates List of (instanceId, transform) pairs
 * @return Number of successful updates
 */
def updateInstanceTransforms(updates: List[(Int, Array[Float])]): Int =
  val successCount = updates.count { case (id, transform) =>
    updateInstanceTransform(id, transform)
  }
  if successCount > 0 then rebuildIAS()
  successCount
```

**`optix-jni/src/main/native/include/OptiXWrapper.h`**

Add method declarations (around line 85, after `getInstanceCount`):

```cpp
    /**
     * Update transform for an existing instance.
     * @param instanceId ID of instance to update
     * @param transform 4x3 row-major transform matrix (12 floats)
     * @return true if update succeeded
     */
    bool updateInstanceTransform(int instanceId, const float* transform);
    
    /**
     * Rebuild Instance Acceleration Structure after transform updates.
     * Must be called before rendering if transforms were modified.
     */
    void rebuildIAS();
```

**`optix-jni/src/main/native/OptiXWrapper.cpp`**

Add implementation (after `getInstanceCount` implementation):

```cpp
bool OptiXWrapper::updateInstanceTransform(int instanceId, const float* transform) {
    if (instanceId < 0 || instanceId >= static_cast<int>(instances_.size())) {
        LOG_ERROR("Invalid instance ID: " << instanceId);
        return false;
    }
    
    auto& instance = instances_[instanceId];
    if (!instance.active) {
        LOG_ERROR("Instance " << instanceId << " is not active");
        return false;
    }
    
    // Update the transform in our instance data
    std::memcpy(instance.transform, transform, 12 * sizeof(float));
    
    // Mark IAS as needing rebuild
    iasNeedsRebuild_ = true;
    
    return true;
}

void OptiXWrapper::rebuildIAS() {
    if (!iasNeedsRebuild_) {
        return;  // No changes, skip rebuild
    }
    
    // Rebuild the instance acceleration structure
    buildIAS();
    iasNeedsRebuild_ = false;
}
```

Add member variable in header (private section):

```cpp
    bool iasNeedsRebuild_ = false;
```

**`optix-jni/src/main/native/JNIBindings.cpp`**

Add JNI binding:

```cpp
JNIEXPORT jboolean JNICALL Java_menger_optix_OptiXRenderer_updateInstanceTransformNative(
    JNIEnv* env, jobject obj, jint instanceId, jfloatArray transform) {
    
    OptiXWrapper* wrapper = getWrapper(env, obj);
    if (!wrapper) return JNI_FALSE;
    
    jfloat* transformData = env->GetFloatArrayElements(transform, nullptr);
    if (!transformData) return JNI_FALSE;
    
    bool result = wrapper->updateInstanceTransform(instanceId, transformData);
    
    env->ReleaseFloatArrayElements(transform, transformData, JNI_ABORT);
    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_menger_optix_OptiXRenderer_rebuildIAS(
    JNIEnv* env, jobject obj) {
    
    OptiXWrapper* wrapper = getWrapper(env, obj);
    if (wrapper) {
        wrapper->rebuildIAS();
    }
}
```

#### Tests to Add

**`optix-jni/src/test/scala/menger/optix/InstanceTransformUpdateSuite.scala`**

```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color
import menger.common.TransformUtil

class InstanceTransformUpdateSuite extends AnyFlatSpec with Matchers with OptiXTestBase:

  "updateInstanceTransform" should "update position of existing instance" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)
    
    val initialTransform = TransformUtil.translation(0f, 0f, 0f)
    val instanceId = renderer.addSphereInstance(initialTransform, Color.White, 1.0f)
    instanceId shouldBe defined
    
    val newTransform = TransformUtil.translation(1f, 2f, 3f)
    val success = renderer.updateInstanceTransform(instanceId.get, newTransform)
    success shouldBe true
  }

  it should "return false for invalid instance ID" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)
    
    val transform = TransformUtil.identity()
    val success = renderer.updateInstanceTransform(999, transform)
    success shouldBe false
  }

  it should "return false for negative instance ID" in withRenderer { renderer =>
    val transform = TransformUtil.identity()
    val success = renderer.updateInstanceTransform(-1, transform)
    success shouldBe false
  }

  it should "reject wrong-sized transform array" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)
    
    val instanceId = renderer.addSphereInstance(TransformUtil.identity(), Color.White, 1.0f)
    
    val wrongSizeTransform = Array(1f, 0f, 0f)  // Too short
    an[IllegalArgumentException] should be thrownBy:
      renderer.updateInstanceTransform(instanceId.get, wrongSizeTransform)
  }

  "updateInstanceTransforms" should "batch update multiple instances" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)
    
    val id1 = renderer.addSphereInstance(TransformUtil.identity(), Color.White, 1.0f).get
    val id2 = renderer.addSphereInstance(TransformUtil.identity(), Color.White, 1.0f).get
    
    val updates = List(
      (id1, TransformUtil.translation(1f, 0f, 0f)),
      (id2, TransformUtil.translation(-1f, 0f, 0f))
    )
    
    val successCount = renderer.updateInstanceTransforms(updates)
    successCount shouldBe 2
  }

  "rebuildIAS" should "not throw when called after transform updates" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)
    
    val id = renderer.addSphereInstance(TransformUtil.identity(), Color.White, 1.0f).get
    renderer.updateInstanceTransform(id, TransformUtil.translation(1f, 0f, 0f))
    
    noException should be thrownBy renderer.rebuildIAS()
  }
```

#### Verification

```bash
sbt "testOnly menger.optix.InstanceTransformUpdateSuite"
```

---

### Step 12.2: Create Animation Data Types

**Status:** Not Started
**Estimate:** 2 hours

Create core data structures for keyframe animation.

#### Files to Create

**`menger-common/src/main/scala/menger/common/animation/Transform3D.scala`**

```scala
package menger.common.animation

import menger.common.Vector

/**
 * 3D transformation with position, rotation, and scale.
 *
 * @param position Translation in world space
 * @param rotation Euler angles in degrees (pitch, yaw, roll)
 * @param scale Non-uniform scale factors
 */
case class Transform3D(
  position: Vector[3] = Vector[3](0f, 0f, 0f),
  rotation: Vector[3] = Vector[3](0f, 0f, 0f),
  scale: Vector[3] = Vector[3](1f, 1f, 1f)
):
  require(scale(0) > 0 && scale(1) > 0 && scale(2) > 0,
    s"Scale must be positive, got $scale")

  /** Convert to 4x3 row-major transform matrix for OptiX. */
  def toMatrix: Array[Float] =
    // Convert rotation from degrees to radians
    val rx = math.toRadians(rotation(0)).toFloat
    val ry = math.toRadians(rotation(1)).toFloat
    val rz = math.toRadians(rotation(2)).toFloat
    
    // Compute rotation matrix components (ZYX order)
    val cx = math.cos(rx).toFloat
    val sx = math.sin(rx).toFloat
    val cy = math.cos(ry).toFloat
    val sy = math.sin(ry).toFloat
    val cz = math.cos(rz).toFloat
    val sz = math.sin(rz).toFloat
    
    // Combined rotation matrix (Rz * Ry * Rx)
    val r00 = cy * cz
    val r01 = cz * sx * sy - cx * sz
    val r02 = cx * cz * sy + sx * sz
    val r10 = cy * sz
    val r11 = cx * cz + sx * sy * sz
    val r12 = -cz * sx + cx * sy * sz
    val r20 = -sy
    val r21 = cy * sx
    val r22 = cx * cy
    
    // Apply scale and build 4x3 matrix
    Array(
      r00 * scale(0), r01 * scale(1), r02 * scale(2), position(0),
      r10 * scale(0), r11 * scale(1), r12 * scale(2), position(1),
      r20 * scale(0), r21 * scale(1), r22 * scale(2), position(2)
    )

object Transform3D:
  val Identity: Transform3D = Transform3D()
  
  def translation(x: Float, y: Float, z: Float): Transform3D =
    Transform3D(position = Vector[3](x, y, z))
  
  def rotation(pitch: Float, yaw: Float, roll: Float): Transform3D =
    Transform3D(rotation = Vector[3](pitch, yaw, roll))
  
  def uniformScale(s: Float): Transform3D =
    Transform3D(scale = Vector[3](s, s, s))
  
  /** Linear interpolation between two transforms. */
  def lerp(a: Transform3D, b: Transform3D, t: Float): Transform3D =
    require(t >= 0f && t <= 1f, s"Interpolation factor t must be in [0, 1], got $t")
    Transform3D(
      position = Vector[3](
        a.position(0) + (b.position(0) - a.position(0)) * t,
        a.position(1) + (b.position(1) - a.position(1)) * t,
        a.position(2) + (b.position(2) - a.position(2)) * t
      ),
      rotation = Vector[3](
        a.rotation(0) + (b.rotation(0) - a.rotation(0)) * t,
        a.rotation(1) + (b.rotation(1) - a.rotation(1)) * t,
        a.rotation(2) + (b.rotation(2) - a.rotation(2)) * t
      ),
      scale = Vector[3](
        a.scale(0) + (b.scale(0) - a.scale(0)) * t,
        a.scale(1) + (b.scale(1) - a.scale(1)) * t,
        a.scale(2) + (b.scale(2) - a.scale(2)) * t
      )
    )
```

**`menger-common/src/main/scala/menger/common/animation/Keyframe.scala`**

```scala
package menger.common.animation

/**
 * A keyframe defining object transform at a specific time.
 *
 * @param time Normalized time [0.0, 1.0] or frame number
 * @param transform The transform at this keyframe
 */
case class Keyframe(
  time: Float,
  transform: Transform3D
):
  require(time >= 0f, s"Keyframe time must be non-negative, got $time")

object Keyframe:
  /** Create keyframe at normalized time with position only. */
  def at(time: Float, x: Float, y: Float, z: Float): Keyframe =
    Keyframe(time, Transform3D.translation(x, y, z))
  
  /** Create keyframe at normalized time with full transform. */
  def at(time: Float, transform: Transform3D): Keyframe =
    Keyframe(time, transform)
  
  /** Create keyframe with identity transform. */
  def identity(time: Float): Keyframe =
    Keyframe(time, Transform3D.Identity)
```

**`menger-common/src/main/scala/menger/common/animation/AnimationTrack.scala`**

```scala
package menger.common.animation

/**
 * Animation track containing keyframes for a single object.
 *
 * @param objectIndex Index of the object in the scene (0-based)
 * @param keyframes List of keyframes, must be sorted by time
 */
case class AnimationTrack(
  objectIndex: Int,
  keyframes: List[Keyframe]
):
  require(objectIndex >= 0, s"Object index must be non-negative, got $objectIndex")
  require(keyframes.nonEmpty, "Animation track must have at least one keyframe")
  require(
    keyframes.sliding(2).forall {
      case List(a, b) => a.time <= b.time
      case _ => true
    },
    "Keyframes must be sorted by time"
  )
  
  /** Get interpolated transform at the given normalized time [0, 1]. */
  def transformAt(normalizedTime: Float): Transform3D =
    val t = normalizedTime.max(0f).min(1f)
    
    // Find surrounding keyframes
    val (before, after) = keyframes.span(_.time <= t)
    
    (before.lastOption, after.headOption) match
      case (Some(a), Some(b)) if a.time != b.time =>
        // Interpolate between keyframes
        val localT = (t - a.time) / (b.time - a.time)
        Transform3D.lerp(a.transform, b.transform, localT)
      case (Some(a), _) =>
        // At or past last keyframe
        a.transform
      case (None, Some(b)) =>
        // Before first keyframe
        b.transform
      case _ =>
        // Should never happen due to require(keyframes.nonEmpty)
        Transform3D.Identity

object AnimationTrack:
  /** Create track from varargs keyframes. */
  def apply(objectIndex: Int, keyframes: Keyframe*): AnimationTrack =
    AnimationTrack(objectIndex, keyframes.toList.sortBy(_.time))
  
  /** Create simple position animation from A to B. */
  def translateFromTo(
    objectIndex: Int,
    from: (Float, Float, Float),
    to: (Float, Float, Float)
  ): AnimationTrack =
    AnimationTrack(
      objectIndex,
      List(
        Keyframe(0f, Transform3D.translation(from._1, from._2, from._3)),
        Keyframe(1f, Transform3D.translation(to._1, to._2, to._3))
      )
    )
  
  /** Create rotation animation. */
  def rotate(
    objectIndex: Int,
    fromAngles: (Float, Float, Float),
    toAngles: (Float, Float, Float)
  ): AnimationTrack =
    AnimationTrack(
      objectIndex,
      List(
        Keyframe(0f, Transform3D.rotation(fromAngles._1, fromAngles._2, fromAngles._3)),
        Keyframe(1f, Transform3D.rotation(toAngles._1, toAngles._2, toAngles._3))
      )
    )
```

**`menger-common/src/main/scala/menger/common/animation/SceneAnimation.scala`**

```scala
package menger.common.animation

/**
 * Complete animation definition for a scene.
 *
 * @param totalFrames Total number of frames to render
 * @param fps Frames per second (for timing calculations)
 * @param tracks Animation tracks for each animated object
 */
case class SceneAnimation(
  totalFrames: Int,
  fps: Float = 30f,
  tracks: List[AnimationTrack] = List.empty
):
  require(totalFrames > 0, s"Total frames must be positive, got $totalFrames")
  require(fps > 0, s"FPS must be positive, got $fps")
  
  /** Duration in seconds. */
  def durationSeconds: Float = totalFrames / fps
  
  /** Get normalized time [0, 1] for a frame number. */
  def normalizedTime(frame: Int): Float =
    if totalFrames <= 1 then 0f
    else frame.toFloat / (totalFrames - 1)
  
  /** Get all transforms for a specific frame. */
  def transformsAtFrame(frame: Int): Map[Int, Transform3D] =
    val t = normalizedTime(frame)
    tracks.map(track => track.objectIndex -> track.transformAt(t)).toMap
  
  /** Check if an object has animation. */
  def hasAnimation(objectIndex: Int): Boolean =
    tracks.exists(_.objectIndex == objectIndex)

object SceneAnimation:
  /** Create animation with single track. */
  def apply(totalFrames: Int, track: AnimationTrack): SceneAnimation =
    SceneAnimation(totalFrames, fps = 30f, tracks = List(track))
  
  /** Create animation with multiple tracks. */
  def apply(totalFrames: Int, fps: Float, tracks: AnimationTrack*): SceneAnimation =
    SceneAnimation(totalFrames, fps, tracks.toList)
  
  /** No animation (single frame). */
  val Static: SceneAnimation = SceneAnimation(totalFrames = 1)
```

#### Tests to Add

**`menger-common/src/test/scala/menger/common/animation/Transform3DSpec.scala`**

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Vector

class Transform3DSpec extends AnyFlatSpec with Matchers:

  "Transform3D" should "have identity default" in:
    val t = Transform3D()
    t.position shouldBe Vector[3](0f, 0f, 0f)
    t.rotation shouldBe Vector[3](0f, 0f, 0f)
    t.scale shouldBe Vector[3](1f, 1f, 1f)

  it should "create translation-only transform" in:
    val t = Transform3D.translation(1f, 2f, 3f)
    t.position shouldBe Vector[3](1f, 2f, 3f)
    t.scale shouldBe Vector[3](1f, 1f, 1f)

  it should "reject non-positive scale" in:
    an[IllegalArgumentException] should be thrownBy:
      Transform3D(scale = Vector[3](0f, 1f, 1f))
    an[IllegalArgumentException] should be thrownBy:
      Transform3D(scale = Vector[3](-1f, 1f, 1f))

  it should "generate identity matrix for identity transform" in:
    val matrix = Transform3D.Identity.toMatrix
    matrix.length shouldBe 12
    // Check diagonal (scale) and translation
    matrix(0) shouldBe 1f +- 0.001f   // scale x
    matrix(5) shouldBe 1f +- 0.001f   // scale y
    matrix(10) shouldBe 1f +- 0.001f  // scale z (actually r22)
    matrix(3) shouldBe 0f             // translate x
    matrix(7) shouldBe 0f             // translate y
    matrix(11) shouldBe 0f            // translate z

  it should "include translation in matrix" in:
    val t = Transform3D.translation(1f, 2f, 3f)
    val matrix = t.toMatrix
    matrix(3) shouldBe 1f   // tx
    matrix(7) shouldBe 2f   // ty
    matrix(11) shouldBe 3f  // tz

  "Transform3D.lerp" should "interpolate position linearly" in:
    val a = Transform3D.translation(0f, 0f, 0f)
    val b = Transform3D.translation(10f, 20f, 30f)
    
    val mid = Transform3D.lerp(a, b, 0.5f)
    mid.position(0) shouldBe 5f +- 0.001f
    mid.position(1) shouldBe 10f +- 0.001f
    mid.position(2) shouldBe 15f +- 0.001f

  it should "return start at t=0" in:
    val a = Transform3D.translation(1f, 2f, 3f)
    val b = Transform3D.translation(10f, 20f, 30f)
    
    val result = Transform3D.lerp(a, b, 0f)
    result.position shouldBe a.position

  it should "return end at t=1" in:
    val a = Transform3D.translation(1f, 2f, 3f)
    val b = Transform3D.translation(10f, 20f, 30f)
    
    val result = Transform3D.lerp(a, b, 1f)
    result.position shouldBe b.position

  it should "interpolate rotation" in:
    val a = Transform3D.rotation(0f, 0f, 0f)
    val b = Transform3D.rotation(90f, 180f, 270f)
    
    val mid = Transform3D.lerp(a, b, 0.5f)
    mid.rotation(0) shouldBe 45f +- 0.001f
    mid.rotation(1) shouldBe 90f +- 0.001f
    mid.rotation(2) shouldBe 135f +- 0.001f

  it should "reject t outside [0, 1]" in:
    val a = Transform3D.Identity
    val b = Transform3D.Identity
    
    an[IllegalArgumentException] should be thrownBy Transform3D.lerp(a, b, -0.1f)
    an[IllegalArgumentException] should be thrownBy Transform3D.lerp(a, b, 1.1f)
```

**`menger-common/src/test/scala/menger/common/animation/AnimationTrackSpec.scala`**

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimationTrackSpec extends AnyFlatSpec with Matchers:

  "AnimationTrack" should "require at least one keyframe" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationTrack(0, List.empty)

  it should "require non-negative object index" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimationTrack(-1, List(Keyframe.identity(0f)))

  it should "sort keyframes by time" in:
    val track = AnimationTrack(0,
      Keyframe.at(1f, 10f, 0f, 0f),
      Keyframe.at(0f, 0f, 0f, 0f),
      Keyframe.at(0.5f, 5f, 0f, 0f)
    )
    track.keyframes.map(_.time) shouldBe List(0f, 0.5f, 1f)

  it should "interpolate between keyframes" in:
    val track = AnimationTrack(0,
      Keyframe.at(0f, 0f, 0f, 0f),
      Keyframe.at(1f, 10f, 0f, 0f)
    )
    
    val mid = track.transformAt(0.5f)
    mid.position(0) shouldBe 5f +- 0.001f

  it should "return first keyframe at t=0" in:
    val track = AnimationTrack(0,
      Keyframe.at(0f, 1f, 2f, 3f),
      Keyframe.at(1f, 10f, 20f, 30f)
    )
    
    val result = track.transformAt(0f)
    result.position(0) shouldBe 1f

  it should "return last keyframe at t=1" in:
    val track = AnimationTrack(0,
      Keyframe.at(0f, 1f, 2f, 3f),
      Keyframe.at(1f, 10f, 20f, 30f)
    )
    
    val result = track.transformAt(1f)
    result.position(0) shouldBe 10f

  it should "clamp time to [0, 1]" in:
    val track = AnimationTrack(0,
      Keyframe.at(0f, 0f, 0f, 0f),
      Keyframe.at(1f, 10f, 0f, 0f)
    )
    
    track.transformAt(-0.5f).position(0) shouldBe 0f
    track.transformAt(1.5f).position(0) shouldBe 10f

  "AnimationTrack.translateFromTo" should "create simple position animation" in:
    val track = AnimationTrack.translateFromTo(0, (0f, 0f, 0f), (10f, 10f, 10f))
    
    track.objectIndex shouldBe 0
    track.keyframes should have size 2
    track.transformAt(0.5f).position(0) shouldBe 5f +- 0.001f
```

**`menger-common/src/test/scala/menger/common/animation/SceneAnimationSpec.scala`**

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneAnimationSpec extends AnyFlatSpec with Matchers:

  "SceneAnimation" should "require positive frame count" in:
    an[IllegalArgumentException] should be thrownBy:
      SceneAnimation(totalFrames = 0)
    an[IllegalArgumentException] should be thrownBy:
      SceneAnimation(totalFrames = -1)

  it should "require positive fps" in:
    an[IllegalArgumentException] should be thrownBy:
      SceneAnimation(totalFrames = 10, fps = 0f)

  it should "calculate duration correctly" in:
    val anim = SceneAnimation(totalFrames = 60, fps = 30f)
    anim.durationSeconds shouldBe 2f +- 0.001f

  it should "calculate normalized time correctly" in:
    val anim = SceneAnimation(totalFrames = 11)  // 0-10, so 11 frames
    
    anim.normalizedTime(0) shouldBe 0f
    anim.normalizedTime(5) shouldBe 0.5f +- 0.001f
    anim.normalizedTime(10) shouldBe 1f

  it should "handle single frame animation" in:
    val anim = SceneAnimation(totalFrames = 1)
    anim.normalizedTime(0) shouldBe 0f

  it should "get transforms at frame" in:
    val track = AnimationTrack.translateFromTo(0, (0f, 0f, 0f), (10f, 0f, 0f))
    val anim = SceneAnimation(totalFrames = 11, fps = 30f, tracks = List(track))
    
    val transforms = anim.transformsAtFrame(5)
    transforms.contains(0) shouldBe true
    transforms(0).position(0) shouldBe 5f +- 0.001f

  it should "check if object has animation" in:
    val track = AnimationTrack.translateFromTo(0, (0f, 0f, 0f), (10f, 0f, 0f))
    val anim = SceneAnimation(totalFrames = 10, fps = 30f, tracks = List(track))
    
    anim.hasAnimation(0) shouldBe true
    anim.hasAnimation(1) shouldBe false

  "SceneAnimation.Static" should "have single frame" in:
    SceneAnimation.Static.totalFrames shouldBe 1
```

#### Verification

```bash
sbt "testOnly menger.common.animation.*"
```

---

### Step 12.3: Create AnimatedOptiXEngine

**Status:** Not Started
**Estimate:** 3 hours

Create the main animation rendering class that extends OptiXEngine with frame-by-frame rendering and image sequence output.

#### Files to Create

**`menger-app/src/main/scala/menger/config/AnimatedOptiXEngineConfig.scala`**

```scala
package menger.config

import menger.common.animation.SceneAnimation

case class AnimatedOptiXEngineConfig(
  base: OptiXEngineConfig,
  animation: SceneAnimation,
  outputPattern: String = "frame_%04d.png"
):
  require(outputPattern.contains("%"), "Output pattern must contain format specifier (e.g. %04d)")
  require(animation.totalFrames > 0, "Animation must have at least one frame")
  
  def formatFramePath(frame: Int): String =
    outputPattern.format(frame)
```

**`menger-app/src/main/scala/menger/engines/AnimatedOptiXEngine.scala`**

```scala
package menger.engines

import java.nio.file.Files
import java.nio.file.Paths

import scala.util.Failure
import scala.util.Success
import scala.util.Try

import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.GL20
import com.typesafe.scalalogging.LazyLogging
import menger.ColorConversions.toCommonColor
import menger.ObjectSpec
import menger.OptiXRenderResources
import menger.ProfilingConfig
import menger.common.ImageSize
import menger.common.TransformUtil
import menger.common.animation.SceneAnimation
import menger.common.animation.Transform3D
import menger.config.AnimatedOptiXEngineConfig
import menger.optix.CameraState
import menger.optix.OptiXRenderer
import menger.optix.OptiXRendererWrapper
import menger.optix.SceneConfigurator

class AnimatedOptiXEngine(config: AnimatedOptiXEngineConfig)(using profilingConfig: ProfilingConfig)
  extends RenderEngine with LazyLogging:

  private val baseConfig = config.base
  private val animation = config.animation
  private val scene = baseConfig.scene
  private val camera = baseConfig.camera
  private val environment = baseConfig.environment
  private val execution = baseConfig.execution

  private val frameCounter = java.util.concurrent.atomic.AtomicInteger(0)
  private var instanceIds: Map[Int, Int] = Map.empty  // objectIndex -> instanceId

  // Composition: renderer and resources
  private val rendererWrapper = OptiXRendererWrapper(execution.maxInstances)
  private lazy val sceneConfigurator = SceneConfigurator(
    Try((_: OptiXRenderer) => ()),  // No default geometry
    camera.position,
    camera.lookAt,
    camera.up,
    environment.plane,
    environment.lights
  )
  private val cameraState = CameraState(camera.position, camera.lookAt, camera.up)
  private val renderResources: OptiXRenderResources = OptiXRenderResources(0, 0)

  override def create(): Unit =
    val result = createAnimatedScene()
    result.recover { case e: Exception =>
      logger.error(s"Failed to create animated OptiX scene: ${e.getMessage}", e)
      Gdx.app.exit()
    }.get

  private def createAnimatedScene(): Try[Unit] = Try:
    logger.info(s"Creating AnimatedOptiXEngine with ${animation.totalFrames} frames, " +
      s"output=${config.outputPattern}")

    val renderer = rendererWrapper.renderer
    sceneConfigurator.configureLights(renderer)
    sceneConfigurator.configurePlane(renderer)
    sceneConfigurator.configureCamera(renderer)

    // Setup objects from objectSpecs
    scene.objectSpecs match
      case Some(specs) if specs.nonEmpty =>
        setupAnimatedObjects(specs, renderer)
      case _ =>
        logger.warn("No objects defined for animation")

    renderer.setRenderConfig(baseConfig.render)
    renderer.setCausticsConfig(baseConfig.caustics)
    environment.planeColor.foreach(sceneConfigurator.setPlaneColor(renderer, _))

    // Disable continuous rendering - we control the frame loop
    Gdx.graphics.setContinuousRendering(true)

  private def setupAnimatedObjects(specs: List[ObjectSpec], renderer: OptiXRenderer): Unit =
    logger.info(s"Setting up ${specs.length} objects for animation")

    specs.zipWithIndex.foreach { case (spec, index) =>
      val (color, ior) = extractMaterialProperties(spec)
      val initialTransform = getTransformForFrame(index, 0)
      val transformArray = initialTransform.toMatrix

      val instanceIdOpt = spec.objectType match
        case "sphere" =>
          renderer.addSphereInstance(transformArray, color, ior)
        case _ =>
          // For other types, we'd need to set base mesh first
          // Simplified: only spheres supported for animation in v1
          logger.warn(s"Object type ${spec.objectType} not yet supported for animation, using sphere")
          renderer.addSphereInstance(transformArray, color, ior)

      instanceIdOpt.foreach { instanceId =>
        instanceIds = instanceIds + (index -> instanceId)
        logger.debug(s"Added animated object $index as instance $instanceId")
      }
    }

  private val defaultColor = menger.common.Color(0.7f, 0.7f, 0.7f)

  private def extractMaterialProperties(spec: ObjectSpec): (menger.common.Color, Float) =
    spec.material match
      case Some(mat) => (mat.color, mat.ior)
      case None => (spec.color.getOrElse(defaultColor), spec.ior)

  private def getTransformForFrame(objectIndex: Int, frame: Int): Transform3D =
    animation.transformsAtFrame(frame).getOrElse(objectIndex, Transform3D.Identity)

  override def render(): Unit =
    val currentFrame = frameCounter.get()

    if currentFrame >= animation.totalFrames then
      logger.info(s"Animation complete: ${animation.totalFrames} frames rendered")
      Gdx.app.exit()
      return

    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT | GL20.GL_DEPTH_BUFFER_BIT)

    val width = Gdx.graphics.getWidth
    val height = Gdx.graphics.getHeight

    if width > 0 && height > 0 then
      // Initialize camera on first render
      if renderResources.currentDimensions.isEmpty then
        cameraState.updateCameraAspectRatio(rendererWrapper.renderer, ImageSize(width, height))

      // Update instance transforms for current frame
      updateTransformsForFrame(currentFrame)

      // Render and save
      val rgbaBytes = rendererWrapper.renderScene(ImageSize(width, height))
      renderResources.renderToScreen(rgbaBytes, width, height)
      saveFrame(currentFrame, width, height)

      // Progress logging every 10%
      val progress = (currentFrame + 1) * 100 / animation.totalFrames
      if currentFrame == 0 || progress % 10 == 0 then
        logger.info(s"Frame ${currentFrame + 1}/${animation.totalFrames} ($progress%)")

    frameCounter.incrementAndGet()

  private def updateTransformsForFrame(frame: Int): Unit =
    val transforms = animation.transformsAtFrame(frame)

    val updates = instanceIds.toList.flatMap { case (objectIndex, instanceId) =>
      transforms.get(objectIndex).map { transform =>
        (instanceId, transform.toMatrix)
      }
    }

    if updates.nonEmpty then
      val successCount = rendererWrapper.renderer.updateInstanceTransforms(updates)
      if successCount != updates.length then
        logger.warn(s"Only $successCount/${updates.length} transform updates succeeded")

  private def saveFrame(frame: Int, width: Int, height: Int): Unit =
    val outputPath = config.formatFramePath(frame)
    Try {
      // Ensure parent directory exists
      val path = Paths.get(outputPath)
      val parent = path.getParent
      if parent != null && !Files.exists(parent) then
        Files.createDirectories(parent)

      ScreenshotFactory.saveScreenshot(outputPath)
    } match
      case Success(_) =>
        logger.debug(s"Saved frame $frame to $outputPath")
      case Failure(e) =>
        logger.error(s"Failed to save frame $frame: ${e.getMessage}")

  override def resize(width: Int, height: Int): Unit = {}

  override def dispose(): Unit =
    logger.debug("Disposing AnimatedOptiXEngine")
    renderResources.dispose()
    rendererWrapper.dispose()

  override def pause(): Unit = {}
  override def resume(): Unit = {}
```

#### Tests to Add

**`menger-app/src/test/scala/menger/engines/AnimatedOptiXEngineConfigSpec.scala`**

```scala
package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.config.AnimatedOptiXEngineConfig
import menger.config.OptiXEngineConfig
import menger.common.animation.SceneAnimation
import menger.common.animation.AnimationTrack

class AnimatedOptiXEngineConfigSpec extends AnyFlatSpec with Matchers:

  val baseConfig = OptiXEngineConfig.default

  "AnimatedOptiXEngineConfig" should "require format specifier in output pattern" in:
    an[IllegalArgumentException] should be thrownBy:
      AnimatedOptiXEngineConfig(baseConfig, SceneAnimation(10), "output.png")

  it should "accept valid format patterns" in:
    noException should be thrownBy:
      AnimatedOptiXEngineConfig(baseConfig, SceneAnimation(10), "frame_%04d.png")

  it should "format frame paths correctly" in:
    val config = AnimatedOptiXEngineConfig(baseConfig, SceneAnimation(100), "output/frame_%04d.png")
    config.formatFramePath(0) shouldBe "output/frame_0000.png"
    config.formatFramePath(42) shouldBe "output/frame_0042.png"
    config.formatFramePath(99) shouldBe "output/frame_0099.png"

  it should "support different format specifiers" in:
    val config1 = AnimatedOptiXEngineConfig(baseConfig, SceneAnimation(10), "f%d.png")
    config1.formatFramePath(5) shouldBe "f5.png"

    val config2 = AnimatedOptiXEngineConfig(baseConfig, SceneAnimation(10), "render_%06d.png")
    config2.formatFramePath(123) shouldBe "render_000123.png"
```

#### Verification

```bash
sbt "testOnly menger.engines.AnimatedOptiXEngineConfigSpec"
```

---

### Step 12.4: Add CLI Options for Animation

**Status:** Not Started
**Estimate:** 2 hours

Add command-line options to trigger animated rendering.

#### Files to Modify

**`menger-app/src/main/scala/menger/MengerCLIOptions.scala`**

Add new options to the Animation group (after line 190, after existing `animate` option):

```scala
  val animateScene: ScallopOption[Boolean] = opt[Boolean](
    required = false, default = Some(false), group = animationGroup,
    descr = "Enable OptiX scene animation mode (requires --optix, --objects)"
  )
  val animationFrames: ScallopOption[Int] = opt[Int](
    required = false, default = Some(60), validate = _ > 0, group = animationGroup,
    descr = "Number of animation frames to render (default: 60)"
  )
  val animationFps: ScallopOption[Float] = opt[Float](
    required = false, default = Some(30f), validate = _ > 0, group = animationGroup,
    descr = "Animation frames per second (default: 30)"
  )
  val animationOutput: ScallopOption[String] = opt[String](
    required = false, default = Some("frame_%04d.png"), group = animationGroup,
    validate = _.contains("%"),
    descr = "Output file pattern with frame number (default: frame_%04d.png)"
  )
```

**`menger-app/src/main/scala/menger/cli/CliValidation.scala`**

Add validation rule for animation options (in the `registerValidationRules()` method):

```scala
  // Animation mode validation
  addValidation {
    if animateScene.isSupplied && !optix.isSupplied then
      Left("--animate-scene requires --optix mode")
    else if animateScene.isSupplied && objects.isEmpty then
      Left("--animate-scene requires --objects to be defined")
    else
      Right(())
  }
```

**`menger-app/src/main/scala/menger/Main.scala`**

Add handler for animation mode (before the existing OptiX mode check):

```scala
    // Animation mode takes precedence
    if options.animateScene() then
      createAnimatedOptiXEngine(options)
    else if options.optix() then
      // ... existing OptiX handling
```

Add new method:

```scala
  private def createAnimatedOptiXEngine(options: MengerCLIOptions)(using ProfilingConfig): RenderEngine =
    import menger.common.animation._
    import menger.config._

    val baseConfig = createOptiXConfig(options)
    
    // Build animation from CLI options
    // For Sprint 12, we create a simple orbit animation around the scene
    // Later, DSL integration (Step 12.5) will allow user-defined animations
    val tracks = options.objects().toList.zipWithIndex.map { case (spec, index) =>
      // Default animation: orbit around Y axis
      AnimationTrack(
        index,
        (0 to 10).map { i =>
          val angle = i * 36f  // 360 degrees / 10 keyframes
          val t = i / 10f
          Keyframe(t, Transform3D(
            position = menger.common.Vector[3](
              (spec.x * math.cos(math.toRadians(angle)) - spec.z * math.sin(math.toRadians(angle))).toFloat,
              spec.y,
              (spec.x * math.sin(math.toRadians(angle)) + spec.z * math.cos(math.toRadians(angle))).toFloat
            ),
            rotation = menger.common.Vector[3](0f, angle, 0f)
          ))
        }.toList
      )
    }

    val animation = SceneAnimation(
      totalFrames = options.animationFrames(),
      fps = options.animationFps(),
      tracks = tracks
    )

    val animConfig = AnimatedOptiXEngineConfig(
      base = baseConfig,
      animation = animation,
      outputPattern = options.animationOutput()
    )

    AnimatedOptiXEngine(animConfig)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/cli/AnimationCLIOptionsSuite.scala`**

```scala
package menger.cli

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.rogach.scallop.exceptions.ScallopException

class AnimationCLIOptionsSuite extends AnyFlatSpec with Matchers:

  "MengerCLIOptions --animate-scene" should "default to false" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    opts.animateScene() shouldBe false

  it should "be settable" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--animate-scene",
      "--objects", "type=sphere:pos=0,0,0:size=1"
    ))
    opts.animateScene() shouldBe true

  it should "require --optix mode" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--animate-scene"))

  it should "require --objects" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq("--optix", "--animate-scene", "--object", "sphere"))

  "MengerCLIOptions --animation-frames" should "default to 60" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    opts.animationFrames() shouldBe 60

  it should "accept positive values" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--animate-scene",
      "--objects", "type=sphere:pos=0,0,0:size=1",
      "--animation-frames", "120"
    ))
    opts.animationFrames() shouldBe 120

  it should "reject non-positive values" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq(
        "--optix", "--animate-scene",
        "--objects", "type=sphere:pos=0,0,0:size=1",
        "--animation-frames", "0"
      ))

  "MengerCLIOptions --animation-fps" should "default to 30" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    opts.animationFps() shouldBe 30f

  it should "accept positive values" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--animate-scene",
      "--objects", "type=sphere:pos=0,0,0:size=1",
      "--animation-fps", "60"
    ))
    opts.animationFps() shouldBe 60f

  "MengerCLIOptions --animation-output" should "default to frame_%04d.png" in:
    val opts = SafeMengerCLIOptions(Seq("--optix", "--object", "sphere"))
    opts.animationOutput() shouldBe "frame_%04d.png"

  it should "require format specifier" in:
    an[ScallopException] should be thrownBy:
      SafeMengerCLIOptions(Seq(
        "--optix", "--animate-scene",
        "--objects", "type=sphere:pos=0,0,0:size=1",
        "--animation-output", "output.png"
      ))

  it should "accept valid patterns" in:
    val opts = SafeMengerCLIOptions(Seq(
      "--optix", "--animate-scene",
      "--objects", "type=sphere:pos=0,0,0:size=1",
      "--animation-output", "renders/scene_%06d.png"
    ))
    opts.animationOutput() shouldBe "renders/scene_%06d.png"
```

#### Verification

```bash
sbt "testOnly menger.cli.AnimationCLIOptionsSuite"
```

---

### Step 12.5: DSL Integration for Animation

**Status:** Not Started
**Estimate:** 2 hours

Extend the Sprint 11 DSL with animation syntax. This step can be done in parallel with or after Sprint 11 implementation.

#### Animation DSL Syntax Design

```scala
// Simple position animation
sphere(Glass).moveTo(0f, 2f, 0f, duration = 1.0f)

// Full keyframe animation
sphere(Glass).animate(
  at(0f) -> position(0f, 0f, 0f),
  at(0.5f) -> position(0f, 2f, 0f).rotation(0f, 180f, 0f),
  at(1f) -> position(0f, 0f, 0f).rotation(0f, 360f, 0f)
)

// Preset animations
cube(Chrome).orbit(center = (0f, 0f, 0f), radius = 2f)
sphere(Glass).bounce(height = 1f)
```

#### Files to Create

**`menger-common/src/main/scala/menger/common/animation/AnimationDSL.scala`**

```scala
package menger.common.animation

import menger.common.Vector

/** DSL for creating animations fluently. */
object AnimationDSL:

  /** Builder for keyframe transforms. */
  case class TransformBuilder(
    pos: Option[Vector[3]] = None,
    rot: Option[Vector[3]] = None,
    scl: Option[Vector[3]] = None
  ):
    def position(x: Float, y: Float, z: Float): TransformBuilder =
      copy(pos = Some(Vector[3](x, y, z)))

    def rotation(pitch: Float, yaw: Float, roll: Float): TransformBuilder =
      copy(rot = Some(Vector[3](pitch, yaw, roll)))

    def scale(x: Float, y: Float, z: Float): TransformBuilder =
      copy(scl = Some(Vector[3](x, y, z)))

    def uniformScale(s: Float): TransformBuilder =
      copy(scl = Some(Vector[3](s, s, s)))

    def toTransform: Transform3D =
      Transform3D(
        position = pos.getOrElse(Vector[3](0f, 0f, 0f)),
        rotation = rot.getOrElse(Vector[3](0f, 0f, 0f)),
        scale = scl.getOrElse(Vector[3](1f, 1f, 1f))
      )

  /** Create a transform builder starting with position. */
  def position(x: Float, y: Float, z: Float): TransformBuilder =
    TransformBuilder().position(x, y, z)

  /** Create a transform builder starting with rotation. */
  def rotation(pitch: Float, yaw: Float, roll: Float): TransformBuilder =
    TransformBuilder().rotation(pitch, yaw, roll)

  /** Create a keyframe at normalized time. */
  def at(time: Float): KeyframeTime = KeyframeTime(time)

  case class KeyframeTime(time: Float):
    def ->(builder: TransformBuilder): Keyframe =
      Keyframe(time, builder.toTransform)

    def ->(transform: Transform3D): Keyframe =
      Keyframe(time, transform)

  /** Create an AnimationTrack from DSL syntax. */
  def track(objectIndex: Int)(keyframes: Keyframe*): AnimationTrack =
    AnimationTrack(objectIndex, keyframes.toList)

  // Preset animation generators

  /** Create orbit animation around Y axis. */
  def orbit(
    objectIndex: Int,
    center: (Float, Float, Float),
    radius: Float,
    keyframeCount: Int = 12
  ): AnimationTrack =
    val keyframes = (0 to keyframeCount).map { i =>
      val t = i.toFloat / keyframeCount
      val angle = t * 360f
      val rad = math.toRadians(angle)
      Keyframe(t, Transform3D(
        position = Vector[3](
          center._1 + radius * math.cos(rad).toFloat,
          center._2,
          center._3 + radius * math.sin(rad).toFloat
        ),
        rotation = Vector[3](0f, angle, 0f)
      ))
    }.toList
    AnimationTrack(objectIndex, keyframes)

  /** Create bounce animation along Y axis. */
  def bounce(
    objectIndex: Int,
    startPos: (Float, Float, Float),
    height: Float,
    keyframeCount: Int = 8
  ): AnimationTrack =
    val keyframes = (0 to keyframeCount).map { i =>
      val t = i.toFloat / keyframeCount
      // Simple sinusoidal bounce
      val y = startPos._2 + height * math.abs(math.sin(t * math.Pi * 2)).toFloat
      Keyframe(t, Transform3D.translation(startPos._1, y, startPos._3))
    }.toList
    AnimationTrack(objectIndex, keyframes)

  /** Create linear movement from A to B. */
  def moveTo(
    objectIndex: Int,
    from: (Float, Float, Float),
    to: (Float, Float, Float)
  ): AnimationTrack =
    AnimationTrack.translateFromTo(objectIndex, from, to)

  /** Create rotation animation. */
  def spin(
    objectIndex: Int,
    axis: Char,
    degrees: Float,
    keyframeCount: Int = 12
  ): AnimationTrack =
    val keyframes = (0 to keyframeCount).map { i =>
      val t = i.toFloat / keyframeCount
      val angle = t * degrees
      val rotation = axis match
        case 'x' | 'X' => Vector[3](angle, 0f, 0f)
        case 'y' | 'Y' => Vector[3](0f, angle, 0f)
        case 'z' | 'Z' => Vector[3](0f, 0f, angle)
        case _ => Vector[3](0f, angle, 0f)  // Default to Y
      Keyframe(t, Transform3D(rotation = rotation))
    }.toList
    AnimationTrack(objectIndex, keyframes)
```

#### Tests to Add

**`menger-common/src/test/scala/menger/common/animation/AnimationDSLSpec.scala`**

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import AnimationDSL._

class AnimationDSLSpec extends AnyFlatSpec with Matchers:

  "TransformBuilder" should "build position-only transforms" in:
    val t = position(1f, 2f, 3f).toTransform
    t.position(0) shouldBe 1f
    t.position(1) shouldBe 2f
    t.position(2) shouldBe 3f
    t.rotation(0) shouldBe 0f  // defaults

  it should "chain position and rotation" in:
    val t = position(1f, 0f, 0f).rotation(0f, 90f, 0f).toTransform
    t.position(0) shouldBe 1f
    t.rotation(1) shouldBe 90f

  "at(time) -> transform" should "create keyframes" in:
    val kf = at(0.5f) -> position(1f, 2f, 3f)
    kf.time shouldBe 0.5f
    kf.transform.position(0) shouldBe 1f

  "track()" should "create AnimationTrack from keyframes" in:
    val t = track(0)(
      at(0f) -> position(0f, 0f, 0f),
      at(0.5f) -> position(5f, 0f, 0f),
      at(1f) -> position(10f, 0f, 0f)
    )
    t.objectIndex shouldBe 0
    t.keyframes should have size 3
    t.transformAt(0.5f).position(0) shouldBe 5f +- 0.001f

  "orbit()" should "create circular animation" in:
    val t = orbit(0, center = (0f, 0f, 0f), radius = 1f, keyframeCount = 4)
    t.keyframes should have size 5  // 0 to 4 inclusive
    
    // At t=0, should be at (1, 0, 0) - cos(0) = 1
    val start = t.transformAt(0f)
    start.position(0) shouldBe 1f +- 0.01f
    start.position(2) shouldBe 0f +- 0.01f
    
    // At t=0.25, should be at (0, 0, 1) - 90 degrees
    val quarter = t.transformAt(0.25f)
    quarter.position(0) shouldBe 0f +- 0.01f
    quarter.position(2) shouldBe 1f +- 0.01f

  "bounce()" should "create vertical bounce animation" in:
    val t = bounce(0, startPos = (0f, 0f, 0f), height = 1f, keyframeCount = 4)
    t.keyframes should have size 5
    
    // At t=0.25, should be near peak (sin(pi/2) = 1)
    val quarter = t.transformAt(0.25f)
    quarter.position(1) shouldBe 1f +- 0.1f

  "moveTo()" should "create linear translation" in:
    val t = moveTo(0, from = (0f, 0f, 0f), to = (10f, 0f, 0f))
    t.transformAt(0f).position(0) shouldBe 0f
    t.transformAt(0.5f).position(0) shouldBe 5f +- 0.001f
    t.transformAt(1f).position(0) shouldBe 10f

  "spin()" should "create rotation animation around Y axis" in:
    val t = spin(0, axis = 'Y', degrees = 360f, keyframeCount = 4)
    t.keyframes should have size 5
    t.transformAt(0f).rotation(1) shouldBe 0f
    t.transformAt(0.5f).rotation(1) shouldBe 180f +- 0.001f
    t.transformAt(1f).rotation(1) shouldBe 360f +- 0.001f
```

#### Verification

```bash
sbt "testOnly menger.common.animation.AnimationDSLSpec"
```

---

### Step 12.6: Example Animated Scenes

**Status:** Not Started
**Estimate:** 1 hour

Create example scenes demonstrating animation capabilities.

#### Files to Create

**`menger-app/src/main/scala/menger/examples/AnimatedScenes.scala`**

```scala
package menger.examples

import menger.common.animation._
import menger.common.animation.AnimationDSL._

/** Collection of example animated scenes for testing and demonstration. */
object AnimatedScenes:

  /** Single sphere orbiting the origin. */
  def orbitingSphere: SceneAnimation =
    SceneAnimation(
      totalFrames = 120,
      fps = 30f,
      tracks = List(
        orbit(0, center = (0f, 0f, 0f), radius = 2f)
      )
    )

  /** Two spheres, one bouncing and one orbiting. */
  def bouncingAndOrbiting: SceneAnimation =
    SceneAnimation(
      totalFrames = 120,
      fps = 30f,
      tracks = List(
        bounce(0, startPos = (0f, 0f, 0f), height = 2f),
        orbit(1, center = (0f, 0f, 0f), radius = 3f)
      )
    )

  /** Sphere moving from left to right. */
  def simpleTranslation: SceneAnimation =
    SceneAnimation(
      totalFrames = 60,
      fps = 30f,
      tracks = List(
        moveTo(0, from = (-3f, 0f, 0f), to = (3f, 0f, 0f))
      )
    )

  /** Spinning cube. */
  def spinningCube: SceneAnimation =
    SceneAnimation(
      totalFrames = 90,
      fps = 30f,
      tracks = List(
        spin(0, axis = 'Y', degrees = 360f)
      )
    )

  /** Complex multi-object scene with various animations. */
  def danceParty: SceneAnimation =
    SceneAnimation(
      totalFrames = 240,
      fps = 30f,
      tracks = List(
        // Center sphere bounces
        bounce(0, startPos = (0f, 0f, 0f), height = 1.5f, keyframeCount = 16),
        // Left sphere orbits
        orbit(1, center = (-2f, 0f, 0f), radius = 1f),
        // Right sphere spins in place
        spin(2, axis = 'Y', degrees = 720f, keyframeCount = 24),
        // Background sphere rises slowly
        moveTo(3, from = (0f, -3f, -5f), to = (0f, 3f, -5f))
      )
    )

  /** Custom keyframe animation demonstrating DSL. */
  def customKeyframes: SceneAnimation =
    val complexTrack = track(0)(
      at(0f)   -> position(0f, 0f, 0f),
      at(0.2f) -> position(2f, 1f, 0f).rotation(0f, 45f, 0f),
      at(0.4f) -> position(2f, 2f, 2f).rotation(0f, 90f, 0f).uniformScale(1.5f),
      at(0.6f) -> position(0f, 2f, 2f).rotation(0f, 135f, 0f),
      at(0.8f) -> position(-2f, 1f, 0f).rotation(0f, 180f, 0f).uniformScale(0.8f),
      at(1f)   -> position(0f, 0f, 0f).rotation(0f, 360f, 0f)
    )

    SceneAnimation(
      totalFrames = 150,
      fps = 30f,
      tracks = List(complexTrack)
    )

  /** Available example scenes by name. */
  val all: Map[String, SceneAnimation] = Map(
    "orbit" -> orbitingSphere,
    "bounce-orbit" -> bouncingAndOrbiting,
    "translate" -> simpleTranslation,
    "spin" -> spinningCube,
    "dance" -> danceParty,
    "custom" -> customKeyframes
  )

  /** Get example scene by name. */
  def get(name: String): Option[SceneAnimation] = all.get(name)
```

#### Tests to Add

**`menger-app/src/test/scala/menger/examples/AnimatedScenesSpec.scala`**

```scala
package menger.examples

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnimatedScenesSpec extends AnyFlatSpec with Matchers:

  "AnimatedScenes.orbitingSphere" should "have expected properties" in:
    val scene = AnimatedScenes.orbitingSphere
    scene.totalFrames shouldBe 120
    scene.fps shouldBe 30f
    scene.tracks should have size 1

  "AnimatedScenes.bouncingAndOrbiting" should "have two tracks" in:
    val scene = AnimatedScenes.bouncingAndOrbiting
    scene.tracks should have size 2

  "AnimatedScenes.danceParty" should "have four tracks" in:
    val scene = AnimatedScenes.danceParty
    scene.tracks should have size 4
    scene.totalFrames shouldBe 240

  "AnimatedScenes.all" should "contain all example scenes" in:
    AnimatedScenes.all should have size 6
    AnimatedScenes.all.keys should contain allOf ("orbit", "bounce-orbit", "translate", "spin", "dance", "custom")

  "AnimatedScenes.get" should "return Some for valid names" in:
    AnimatedScenes.get("orbit") shouldBe defined
    AnimatedScenes.get("invalid") shouldBe None

  "All example scenes" should "be valid and executable" in:
    AnimatedScenes.all.values.foreach { scene =>
      scene.totalFrames should be > 0
      scene.fps should be > 0f
      // Verify we can get transforms for any frame
      noException should be thrownBy:
        scene.transformsAtFrame(0)
        scene.transformsAtFrame(scene.totalFrames / 2)
        scene.transformsAtFrame(scene.totalFrames - 1)
    }
```

#### Verification

```bash
sbt "testOnly menger.examples.AnimatedScenesSpec"
```

---

### Step 12.7: Integration Tests

**Status:** Not Started
**Estimate:** 1.5 hours

Create integration tests that verify the full animation pipeline works correctly.

#### Files to Create

**`menger-app/src/test/scala/menger/engines/AnimatedOptiXEngineIntegrationSpec.scala`**

```scala
package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.config.AnimatedOptiXEngineConfig
import menger.config.OptiXEngineConfig
import menger.common.animation._
import menger.common.animation.AnimationDSL._
import menger.ObjectSpec
import menger.common.Color
import menger.ProfilingConfig

class AnimatedOptiXEngineIntegrationSpec extends AnyFlatSpec with Matchers:

  // These tests verify configuration and setup without requiring OptiX hardware

  "AnimatedOptiXEngineConfig" should "accept valid animation configuration" in:
    val animation = SceneAnimation(
      totalFrames = 30,
      fps = 30f,
      tracks = List(AnimationTrack.translateFromTo(0, (0f, 0f, 0f), (1f, 0f, 0f)))
    )

    val baseConfig = OptiXEngineConfig.default.copy(
      scene = OptiXEngineConfig.default.scene.copy(
        objectSpecs = Some(List(
          ObjectSpec(
            objectType = "sphere",
            x = 0f, y = 0f, z = 0f,
            size = 1f,
            color = Some(Color.White),
            ior = 1.5f
          )
        ))
      )
    )

    noException should be thrownBy:
      AnimatedOptiXEngineConfig(
        base = baseConfig,
        animation = animation,
        outputPattern = "test_%04d.png"
      )

  it should "validate output pattern format" in:
    val animation = SceneAnimation(totalFrames = 10)

    an[IllegalArgumentException] should be thrownBy:
      AnimatedOptiXEngineConfig(
        base = OptiXEngineConfig.default,
        animation = animation,
        outputPattern = "no_format.png"
      )

  "SceneAnimation with tracks" should "provide transforms for each frame" in:
    val track = track(0)(
      at(0f) -> position(0f, 0f, 0f),
      at(1f) -> position(10f, 0f, 0f)
    )

    val animation = SceneAnimation(totalFrames = 11, tracks = List(track))

    // Frame 0 should be at start
    animation.transformsAtFrame(0)(0).position(0) shouldBe 0f +- 0.001f

    // Frame 5 should be midway
    animation.transformsAtFrame(5)(0).position(0) shouldBe 5f +- 0.001f

    // Frame 10 should be at end
    animation.transformsAtFrame(10)(0).position(0) shouldBe 10f +- 0.001f

  "Multiple animation tracks" should "animate different objects independently" in:
    val track1 = AnimationTrack.translateFromTo(0, (0f, 0f, 0f), (5f, 0f, 0f))
    val track2 = AnimationTrack.translateFromTo(1, (0f, 0f, 0f), (0f, 5f, 0f))

    val animation = SceneAnimation(totalFrames = 11, tracks = List(track1, track2))

    val midFrame = animation.transformsAtFrame(5)
    midFrame(0).position(0) shouldBe 2.5f +- 0.001f  // X movement
    midFrame(0).position(1) shouldBe 0f             // No Y movement for object 0
    midFrame(1).position(0) shouldBe 0f             // No X movement for object 1
    midFrame(1).position(1) shouldBe 2.5f +- 0.001f // Y movement
```

**`optix-jni/src/test/scala/menger/optix/TransformUpdateIntegrationSuite.scala`**

```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color
import menger.common.TransformUtil
import menger.common.animation.Transform3D

class TransformUpdateIntegrationSuite extends AnyFlatSpec with Matchers with OptiXTestBase:

  "Transform3D.toMatrix" should "produce valid OptiX transform matrices" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)

    // Create instance with identity transform
    val identity = Transform3D.Identity.toMatrix
    identity.length shouldBe 12

    val instanceId = renderer.addSphereInstance(identity, Color.White, 1.0f)
    instanceId shouldBe defined
  }

  it should "support translation transforms" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)

    val transform = Transform3D.translation(1f, 2f, 3f).toMatrix
    val instanceId = renderer.addSphereInstance(transform, Color.White, 1.0f)

    instanceId shouldBe defined

    // Verify we can update to a new position
    val newTransform = Transform3D.translation(4f, 5f, 6f).toMatrix
    val success = renderer.updateInstanceTransform(instanceId.get, newTransform)
    success shouldBe true
  }

  it should "support rotation transforms" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)

    val transform = Transform3D.rotation(45f, 90f, 0f).toMatrix
    val instanceId = renderer.addSphereInstance(transform, Color.White, 1.0f)

    instanceId shouldBe defined
  }

  it should "support combined transforms" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)

    val transform = Transform3D(
      position = menger.common.Vector[3](1f, 0f, 0f),
      rotation = menger.common.Vector[3](0f, 45f, 0f),
      scale = menger.common.Vector[3](2f, 2f, 2f)
    ).toMatrix

    val instanceId = renderer.addSphereInstance(transform, Color.White, 1.0f)
    instanceId shouldBe defined
  }

  "Animation frame updates" should "work for multiple instances" in withRenderer { renderer =>
    renderer.setSphere(menger.common.Vector[3](0f, 0f, 0f), 0.5f)
    renderer.setIASMode(true)

    // Create 3 instances
    val id1 = renderer.addSphereInstance(Transform3D.Identity.toMatrix, Color.Red, 1.0f).get
    val id2 = renderer.addSphereInstance(Transform3D.Identity.toMatrix, Color.Green, 1.0f).get
    val id3 = renderer.addSphereInstance(Transform3D.Identity.toMatrix, Color.Blue, 1.0f).get

    // Batch update all transforms
    val updates = List(
      (id1, Transform3D.translation(-1f, 0f, 0f).toMatrix),
      (id2, Transform3D.translation(0f, 0f, 0f).toMatrix),
      (id3, Transform3D.translation(1f, 0f, 0f).toMatrix)
    )

    val successCount = renderer.updateInstanceTransforms(updates)
    successCount shouldBe 3
  }
```

#### Verification

```bash
sbt "testOnly menger.engines.AnimatedOptiXEngineIntegrationSpec menger.optix.TransformUpdateIntegrationSuite"
```

---

### Step 12.8: Documentation Updates

**Status:** Not Started
**Estimate:** 1 hour

Update documentation to reflect new animation capabilities.

#### Files to Modify

**`CHANGELOG.md`**

Add entry:

```markdown
## [Unreleased]

### Added
- **Object Animation Foundation (Sprint 12)**
  - Frame-by-frame object transform animation in OptiX mode
  - Keyframe-based animation with linear interpolation
  - Position, rotation, and scale animation support
  - Image sequence output (PNG) with configurable naming
  - CLI options: `--animate-scene`, `--animation-frames`, `--animation-fps`, `--animation-output`
  - Animation DSL: `orbit()`, `bounce()`, `moveTo()`, `spin()` presets
  - Custom keyframe animations via `track()` and `at()` DSL
  - Example scenes: orbit, bounce-orbit, translate, spin, dance, custom
  - New JNI methods: `updateInstanceTransform()`, `rebuildIAS()`
```

**`README.md`**

Add section after "Features":

```markdown
### Animation

Menger supports animated scene rendering in OptiX mode:

```bash
# Render 120 frames of a bouncing sphere
menger --optix --animate-scene \
  --objects "type=sphere:pos=0,1,0:size=1:ior=1.5" \
  --animation-frames 120 \
  --animation-output "frames/bounce_%04d.png"

# Convert to video with ffmpeg
ffmpeg -framerate 30 -i frames/bounce_%04d.png -c:v libx264 -pix_fmt yuv420p bounce.mp4
```

Animation features:
- Keyframe-based animation with linear interpolation
- Position, rotation, and scale transforms
- Preset animations: orbit, bounce, translate, spin
- Custom keyframe DSL for complex animations
```

**`TODO.md`**

Move animation items from "Planned" to "Completed":

```markdown
## Completed in Sprint 12
- [x] Object animation foundation
- [x] Keyframe animation system
- [x] Linear interpolation
- [x] Image sequence output
- [x] Animation CLI options
- [x] Animation DSL presets

## Future (Sprint 13+)
- [ ] Easing functions (ease-in-out, cubic, bounce)
- [ ] Camera animation (path following)
- [ ] Light animation
- [ ] Quaternion SLERP for rotation
- [ ] Animation preview mode
- [ ] Direct video output (MP4/WebM)
```

---

## Summary

| Step | Task | Estimate | Status |
|------|------|----------|--------|
| 12.1 | JNI updateInstanceTransform | 2.5h | Not Started |
| 12.2 | Animation Data Types | 2h | Not Started |
| 12.3 | AnimatedOptiXEngine | 3h | Not Started |
| 12.4 | CLI Options | 2h | Not Started |
| 12.5 | DSL Integration | 2h | Not Started |
| 12.6 | Example Scenes | 1h | Not Started |
| 12.7 | Integration Tests | 1.5h | Not Started |
| 12.8 | Documentation | 1h | Not Started |
| **Total** | | **15h** | |

---

## Notes

### Implementation Order

Recommended order for minimal dependencies:

1. **Step 12.2** (Animation Data Types) - Foundation, no dependencies
2. **Step 12.1** (JNI method) - Requires OptiX but independent of animation types
3. **Step 12.3** (AnimatedOptiXEngine) - Depends on 12.1 and 12.2
4. **Step 12.4** (CLI Options) - Depends on 12.3
5. **Step 12.5** (DSL Integration) - Can be done in parallel after 12.2
6. **Step 12.6** (Examples) - Depends on 12.5
7. **Step 12.7** (Integration Tests) - After 12.3
8. **Step 12.8** (Documentation) - Last

### Testing Strategy

- **Unit tests**: Each component in isolation (Transform3D, Keyframe, AnimationTrack)
- **Integration tests**: Full pipeline without hardware (config validation, frame calculations)
- **Hardware tests**: With OptiX renderer (transform updates, IAS rebuild)
- **Visual tests**: Manual verification of output images

### Known Limitations

1. **Spheres only in v1**: Triangle mesh animation requires additional work for IAS mode
2. **Linear interpolation only**: Easing functions deferred to Sprint 13
3. **No camera animation**: Fixed camera; camera paths deferred
4. **No preview mode**: Renders directly to disk; interactive preview deferred

### Performance Considerations

- `updateInstanceTransform()` is O(1) per instance
- `rebuildIAS()` is O(n) where n = instance count; batching updates minimizes rebuilds
- Frame rendering time dominated by ray tracing, not transform updates
- Disk I/O for PNG saving may become bottleneck for high frame counts

### Sprint 11 Integration

If Sprint 11 (Scala DSL) is complete:

```scala
// In DSL scene definition
scene {
  val sphere1 = sphere(Glass).at(0, 0, 0)
  val sphere2 = sphere(Chrome).at(2, 0, 0)
  
  animate(120.frames, 30.fps) {
    sphere1.orbit(radius = 2f)
    sphere2.bounce(height = 1f)
  }
}
```

If Sprint 11 is not complete, animations are defined programmatically via `SceneAnimation` case classes.
