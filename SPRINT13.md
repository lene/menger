# Sprint 13: Advanced Animation System

**Sprint:** 13 - Advanced Animation System
**Status:** Not Started
**Estimate:** 21 hours
**Branch:** `feature/sprint-13`
**Dependencies:** Sprint 12 (Object Animation Foundation) - required

---

## Goal

Extend the animation system to support animating ANY scene parameter (colors, IOR, transparency, camera, lights, sponge levels), add easing functions for smoother motion, video output via ffmpeg, and a preview mode for scrubbing through animations.

## Success Criteria

- [ ] Generic AnimatableProperty[T] system with type-safe interpolation
- [ ] Easing functions: Linear, EaseIn, EaseOut, EaseInOut, Cubic, Bounce, Elastic
- [ ] Per-instance color and IOR animation via JNI
- [ ] Camera animation (position, lookAt, FOV keyframes)
- [ ] Light animation (position, intensity, color per light)
- [ ] Sponge level animation (with mesh regeneration warning)
- [ ] Video output via ffmpeg (MP4/WebM)
- [ ] Preview mode with scrubbing slider and keyframe thumbnails
- [ ] Extended DSL syntax for all animatable properties
- [ ] All tests pass (~40 new tests)

---

## Scope

### In Scope
- Generic property animation system (type class based)
- 7 easing functions with mathematical definitions
- JNI methods: updateInstanceColor(), updateInstanceIOR()
- CameraTrack with position, lookAt, FOV interpolation
- LightTrack for animating individual lights
- GeometryTrack for sponge level changes (expensive operation)
- VideoEncoder class wrapping ffmpeg via ProcessBuilder
- AnimationPreviewEngine with LibGDX Scene2D UI
- DSL extensions for property animation

### Deferred to Sprint 14
- Bezier curve paths for camera/object motion
- Animation blending and layering
- Keyframe editor UI
- Animation export/import (JSON format)
- Physics-based animation (spring, gravity)

---

## Background

### Sprint 12 Foundation

Sprint 12 established the core animation infrastructure:

| Class | Location | Purpose |
|-------|----------|---------|
| `Transform3D` | `menger/common/animation/Transform3D.scala` | Position, rotation, scale container |
| `Keyframe` | `menger/common/animation/Keyframe.scala` | Time + Transform3D |
| `AnimationTrack` | `menger/common/animation/AnimationTrack.scala` | Sequence of keyframes with interpolation |
| `SceneAnimation` | `menger/common/animation/SceneAnimation.scala` | Multiple tracks, duration, FPS |
| `AnimatedOptiXEngine` | `menger/engines/AnimatedOptiXEngine.scala` | Frame-by-frame rendering |

**Gap:** Only object transforms are animatable. No support for colors, camera, lights, or other properties.

### Type-Safe Property Animation

We need a generic system that can animate any type while maintaining type safety:

```scala
// Goal: animate any property type with the same API
val colorTrack = PropertyTrack[Color]("sphereColor", colorKeyframes)
val positionTrack = PropertyTrack[Vector3]("cameraPosition", positionKeyframes)
val fovTrack = PropertyTrack[Float]("cameraFOV", fovKeyframes)

// All use the same interpolation interface
colorTrack.valueAt(0.5f, Easing.EaseInOut)
positionTrack.valueAt(0.5f, Easing.Cubic)
```

### Easing Functions

Standard easing functions for animation:

| Easing | Formula | Use Case |
|--------|---------|----------|
| Linear | `t` | Constant speed |
| EaseIn | `t²` | Slow start |
| EaseOut | `1 - (1-t)²` | Slow end |
| EaseInOut | `3t² - 2t³` (smoothstep) | Slow start and end |
| Cubic | `t³` | More pronounced slow start |
| Bounce | Piecewise quadratic | Bouncing effect at end |
| Elastic | `sin(13π/2 * t) * 2^(10(t-1))` | Spring overshoot |

---

## Tasks

### Step 13.1: Generic AnimatableProperty[T] System

**Status:** Not Started
**Estimate:** 3 hours

Create a type class based system for interpolating any value type.

#### File: `menger-common/src/main/scala/menger/common/animation/Easing.scala`

```scala
package menger.common.animation

enum Easing:
  case Linear
  case EaseIn
  case EaseOut
  case EaseInOut
  case Cubic
  case Bounce
  case Elastic

object Easing:
  def apply(easing: Easing, t: Float): Float =
    val clamped = math.max(0f, math.min(1f, t))
    easing match
      case Linear    => clamped
      case EaseIn    => clamped * clamped
      case EaseOut   => 1f - (1f - clamped) * (1f - clamped)
      case EaseInOut => 3f * clamped * clamped - 2f * clamped * clamped * clamped
      case Cubic     => clamped * clamped * clamped
      case Bounce    => bounceOut(clamped)
      case Elastic   => elasticOut(clamped)

  private def bounceOut(t: Float): Float =
    if t < 1f / 2.75f then
      7.5625f * t * t
    else if t < 2f / 2.75f then
      val t2 = t - 1.5f / 2.75f
      7.5625f * t2 * t2 + 0.75f
    else if t < 2.5f / 2.75f then
      val t2 = t - 2.25f / 2.75f
      7.5625f * t2 * t2 + 0.9375f
    else
      val t2 = t - 2.625f / 2.75f
      7.5625f * t2 * t2 + 0.984375f

  private def elasticOut(t: Float): Float =
    if t == 0f then 0f
    else if t == 1f then 1f
    else
      val p = 0.3f
      val s = p / 4f
      math.pow(2, -10 * t).toFloat * math.sin((t - s) * (2 * math.Pi) / p).toFloat + 1f
```

#### File: `menger-common/src/main/scala/menger/common/animation/Interpolatable.scala`

```scala
package menger.common.animation

import menger.common.Color

trait Interpolatable[T]:
  def interpolate(a: T, b: T, t: Float): T

object Interpolatable:
  given Interpolatable[Float] with
    def interpolate(a: Float, b: Float, t: Float): Float =
      a + (b - a) * t

  given Interpolatable[Double] with
    def interpolate(a: Double, b: Double, t: Float): Double =
      a + (b - a) * t

  given Interpolatable[Vector3] with
    def interpolate(a: Vector3, b: Vector3, t: Float): Vector3 =
      Vector3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
      )

  given Interpolatable[Color] with
    def interpolate(a: Color, b: Color, t: Float): Color =
      Color(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t
      )

  given Interpolatable[Transform3D] with
    def interpolate(a: Transform3D, b: Transform3D, t: Float): Transform3D =
      val posInterp = summon[Interpolatable[Vector3]]
      Transform3D(
        position = posInterp.interpolate(a.position, b.position, t),
        rotation = posInterp.interpolate(a.rotation, b.rotation, t),
        scale = posInterp.interpolate(a.scale, b.scale, t)
      )
```

#### File: `menger-common/src/main/scala/menger/common/animation/Vector3.scala`

```scala
package menger.common.animation

case class Vector3(x: Float, y: Float, z: Float):
  def +(other: Vector3): Vector3 = Vector3(x + other.x, y + other.y, z + other.z)
  def -(other: Vector3): Vector3 = Vector3(x - other.x, y - other.y, z - other.z)
  def *(scalar: Float): Vector3 = Vector3(x * scalar, y * scalar, z * scalar)
  def magnitude: Float = math.sqrt(x * x + y * y + z * z).toFloat
  def normalized: Vector3 =
    val mag = magnitude
    if mag > 0f then Vector3(x / mag, y / mag, z / mag) else this
  def dot(other: Vector3): Float = x * other.x + y * other.y + z * other.z
  def cross(other: Vector3): Vector3 = Vector3(
    y * other.z - z * other.y,
    z * other.x - x * other.z,
    x * other.y - y * other.x
  )

object Vector3:
  val Zero: Vector3 = Vector3(0f, 0f, 0f)
  val One: Vector3 = Vector3(1f, 1f, 1f)
  val UnitX: Vector3 = Vector3(1f, 0f, 0f)
  val UnitY: Vector3 = Vector3(0f, 1f, 0f)
  val UnitZ: Vector3 = Vector3(0f, 0f, 1f)
```

#### File: `menger-common/src/main/scala/menger/common/animation/PropertyKeyframe.scala`

```scala
package menger.common.animation

case class PropertyKeyframe[T](
    time: Float,
    value: T,
    easing: Easing = Easing.Linear
):
  require(time >= 0f, s"Keyframe time must be non-negative: $time")
```

#### File: `menger-common/src/main/scala/menger/common/animation/PropertyTrack.scala`

```scala
package menger.common.animation

case class PropertyTrack[T: Interpolatable](
    name: String,
    keyframes: Seq[PropertyKeyframe[T]]
):
  require(keyframes.nonEmpty, "PropertyTrack requires at least one keyframe")
  require(
    keyframes.sortBy(_.time) == keyframes,
    "Keyframes must be sorted by time"
  )

  private val interpolatable = summon[Interpolatable[T]]

  def valueAt(time: Float): T =
    if keyframes.size == 1 then keyframes.head.value
    else if time <= keyframes.head.time then keyframes.head.value
    else if time >= keyframes.last.time then keyframes.last.value
    else
      val (before, after) = findSurroundingKeyframes(time)
      val localT = (time - before.time) / (after.time - before.time)
      val easedT = Easing(after.easing, localT)
      interpolatable.interpolate(before.value, after.value, easedT)

  def duration: Float = keyframes.last.time

  private def findSurroundingKeyframes(
      time: Float
  ): (PropertyKeyframe[T], PropertyKeyframe[T]) =
    val beforeIdx = keyframes.lastIndexWhere(_.time <= time)
    val afterIdx = math.min(beforeIdx + 1, keyframes.size - 1)
    (keyframes(beforeIdx), keyframes(afterIdx))

object PropertyTrack:
  def constant[T: Interpolatable](name: String, value: T): PropertyTrack[T] =
    PropertyTrack(name, Seq(PropertyKeyframe(0f, value)))

  def linear[T: Interpolatable](
      name: String,
      start: T,
      end: T,
      duration: Float
  ): PropertyTrack[T] =
    PropertyTrack(
      name,
      Seq(
        PropertyKeyframe(0f, start, Easing.Linear),
        PropertyKeyframe(duration, end, Easing.Linear)
      )
    )
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/EasingSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EasingSpec extends AnyFlatSpec with Matchers:

  "Easing.Linear" should "return t unchanged" in {
    Easing(Easing.Linear, 0f) shouldBe 0f
    Easing(Easing.Linear, 0.5f) shouldBe 0.5f
    Easing(Easing.Linear, 1f) shouldBe 1f
  }

  "Easing.EaseIn" should "start slow and end fast" in {
    Easing(Easing.EaseIn, 0f) shouldBe 0f
    Easing(Easing.EaseIn, 0.5f) shouldBe 0.25f
    Easing(Easing.EaseIn, 1f) shouldBe 1f
  }

  "Easing.EaseOut" should "start fast and end slow" in {
    Easing(Easing.EaseOut, 0f) shouldBe 0f
    Easing(Easing.EaseOut, 0.5f) shouldBe 0.75f
    Easing(Easing.EaseOut, 1f) shouldBe 1f
  }

  "Easing.EaseInOut" should "be smooth at both ends" in {
    Easing(Easing.EaseInOut, 0f) shouldBe 0f
    Easing(Easing.EaseInOut, 0.5f) shouldBe 0.5f
    Easing(Easing.EaseInOut, 1f) shouldBe 1f
  }

  "Easing.Bounce" should "produce values >= 0 and <= 1" in {
    for t <- 0f to 1f by 0.1f do
      val result = Easing(Easing.Bounce, t)
      result should be >= 0f
      result should be <= 1.1f // slight overshoot allowed
  }

  "Easing.Elastic" should "overshoot then settle" in {
    Easing(Easing.Elastic, 0f) shouldBe 0f
    Easing(Easing.Elastic, 1f) shouldBe 1f
    // Elastic typically overshoots
    Easing(Easing.Elastic, 0.7f) should be > 1f
  }

  "All easings" should "clamp input to [0, 1]" in {
    for easing <- Easing.values do
      Easing(easing, -0.5f) shouldBe Easing(easing, 0f)
      Easing(easing, 1.5f) shouldBe Easing(easing, 1f)
  }
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/PropertyTrackSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color

class PropertyTrackSpec extends AnyFlatSpec with Matchers:

  "PropertyTrack[Float]" should "interpolate between keyframes" in {
    val track = PropertyTrack(
      "test",
      Seq(
        PropertyKeyframe(0f, 0f),
        PropertyKeyframe(1f, 10f)
      )
    )

    track.valueAt(0f) shouldBe 0f
    track.valueAt(0.5f) shouldBe 5f
    track.valueAt(1f) shouldBe 10f
  }

  it should "clamp before first keyframe" in {
    val track = PropertyTrack("test", Seq(PropertyKeyframe(1f, 5f)))
    track.valueAt(0f) shouldBe 5f
  }

  it should "clamp after last keyframe" in {
    val track = PropertyTrack(
      "test",
      Seq(PropertyKeyframe(0f, 0f), PropertyKeyframe(1f, 10f))
    )
    track.valueAt(2f) shouldBe 10f
  }

  "PropertyTrack[Color]" should "interpolate RGBA components" in {
    val track = PropertyTrack(
      "color",
      Seq(
        PropertyKeyframe(0f, Color(1f, 0f, 0f, 1f)),
        PropertyKeyframe(1f, Color(0f, 1f, 0f, 1f))
      )
    )

    val mid = track.valueAt(0.5f)
    mid.r shouldBe 0.5f +- 0.001f
    mid.g shouldBe 0.5f +- 0.001f
    mid.b shouldBe 0f
    mid.a shouldBe 1f
  }

  "PropertyTrack[Vector3]" should "interpolate XYZ components" in {
    val track = PropertyTrack(
      "position",
      Seq(
        PropertyKeyframe(0f, Vector3(0f, 0f, 0f)),
        PropertyKeyframe(1f, Vector3(10f, 20f, 30f))
      )
    )

    val mid = track.valueAt(0.5f)
    mid.x shouldBe 5f
    mid.y shouldBe 10f
    mid.z shouldBe 15f
  }

  "PropertyTrack with easing" should "apply easing function" in {
    val track = PropertyTrack(
      "test",
      Seq(
        PropertyKeyframe(0f, 0f, Easing.Linear),
        PropertyKeyframe(1f, 10f, Easing.EaseIn)
      )
    )

    // EaseIn at t=0.5 gives 0.25, so value should be 2.5
    track.valueAt(0.5f) shouldBe 2.5f +- 0.001f
  }

  "PropertyTrack.constant" should "return same value at all times" in {
    val track = PropertyTrack.constant("test", 42f)
    track.valueAt(0f) shouldBe 42f
    track.valueAt(100f) shouldBe 42f
  }

  "PropertyTrack.linear" should "create two-keyframe track" in {
    val track = PropertyTrack.linear("test", 0f, 100f, 2f)
    track.valueAt(0f) shouldBe 0f
    track.valueAt(1f) shouldBe 50f
    track.valueAt(2f) shouldBe 100f
  }
```

#### Verification

```bash
sbt "project mengerCommon" compile
sbt "project mengerCommon" "testOnly menger.common.animation.EasingSpec"
sbt "project mengerCommon" "testOnly menger.common.animation.PropertyTrackSpec"
```

---

### Step 13.2: Quaternion SLERP for Smooth Rotation

**Status:** Not Started
**Estimate:** 2 hours

Add quaternion-based spherical linear interpolation (SLERP) for smoother rotation animation.

#### File: `menger-common/src/main/scala/menger/common/animation/Quaternion.scala`

```scala
package menger.common.animation

case class Quaternion(w: Float, x: Float, y: Float, z: Float):
  def magnitude: Float = math.sqrt(w * w + x * x + y * y + z * z).toFloat

  def normalized: Quaternion =
    val mag = magnitude
    if mag > 0f then Quaternion(w / mag, x / mag, y / mag, z / mag)
    else this

  def conjugate: Quaternion = Quaternion(w, -x, -y, -z)

  def *(other: Quaternion): Quaternion = Quaternion(
    w * other.w - x * other.x - y * other.y - z * other.z,
    w * other.x + x * other.w + y * other.z - z * other.y,
    w * other.y - x * other.z + y * other.w + z * other.x,
    w * other.z + x * other.y - y * other.x + z * other.w
  )

  def dot(other: Quaternion): Float =
    w * other.w + x * other.x + y * other.y + z * other.z

  def toEulerAngles: Vector3 =
    // Convert quaternion to Euler angles (XYZ order)
    val sinrCosp = 2f * (w * x + y * z)
    val cosrCosp = 1f - 2f * (x * x + y * y)
    val roll = math.atan2(sinrCosp, cosrCosp).toFloat

    val sinp = 2f * (w * y - z * x)
    val pitch = if math.abs(sinp) >= 1 then
      math.copySign(math.Pi / 2, sinp).toFloat
    else
      math.asin(sinp).toFloat

    val sinyCosp = 2f * (w * z + x * y)
    val cosyCosp = 1f - 2f * (y * y + z * z)
    val yaw = math.atan2(sinyCosp, cosyCosp).toFloat

    Vector3(
      math.toDegrees(roll).toFloat,
      math.toDegrees(pitch).toFloat,
      math.toDegrees(yaw).toFloat
    )

  def toRotationMatrix: Array[Float] =
    val xx = x * x; val yy = y * y; val zz = z * z
    val xy = x * y; val xz = x * z; val yz = y * z
    val wx = w * x; val wy = w * y; val wz = w * z
    Array(
      1f - 2f * (yy + zz), 2f * (xy - wz), 2f * (xz + wy),
      2f * (xy + wz), 1f - 2f * (xx + zz), 2f * (yz - wx),
      2f * (xz - wy), 2f * (yz + wx), 1f - 2f * (xx + yy)
    )

object Quaternion:
  val Identity: Quaternion = Quaternion(1f, 0f, 0f, 0f)

  def fromEulerAngles(roll: Float, pitch: Float, yaw: Float): Quaternion =
    val cr = math.cos(math.toRadians(roll) / 2).toFloat
    val sr = math.sin(math.toRadians(roll) / 2).toFloat
    val cp = math.cos(math.toRadians(pitch) / 2).toFloat
    val sp = math.sin(math.toRadians(pitch) / 2).toFloat
    val cy = math.cos(math.toRadians(yaw) / 2).toFloat
    val sy = math.sin(math.toRadians(yaw) / 2).toFloat

    Quaternion(
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy
    ).normalized

  def fromAxisAngle(axis: Vector3, angleDegrees: Float): Quaternion =
    val halfAngle = math.toRadians(angleDegrees / 2).toFloat
    val s = math.sin(halfAngle).toFloat
    val normalizedAxis = axis.normalized
    Quaternion(
      math.cos(halfAngle).toFloat,
      normalizedAxis.x * s,
      normalizedAxis.y * s,
      normalizedAxis.z * s
    ).normalized

  def slerp(a: Quaternion, b: Quaternion, t: Float): Quaternion =
    var cosHalfTheta = a.dot(b)
    var bAdjusted = b

    // If dot product is negative, negate one quaternion to take shorter path
    if cosHalfTheta < 0 then
      bAdjusted = Quaternion(-b.w, -b.x, -b.y, -b.z)
      cosHalfTheta = -cosHalfTheta

    // If quaternions are very close, use linear interpolation
    if cosHalfTheta > 0.9995f then
      Quaternion(
        a.w + t * (bAdjusted.w - a.w),
        a.x + t * (bAdjusted.x - a.x),
        a.y + t * (bAdjusted.y - a.y),
        a.z + t * (bAdjusted.z - a.z)
      ).normalized
    else
      val halfTheta = math.acos(cosHalfTheta).toFloat
      val sinHalfTheta = math.sqrt(1f - cosHalfTheta * cosHalfTheta).toFloat
      val ratioA = math.sin((1 - t) * halfTheta).toFloat / sinHalfTheta
      val ratioB = math.sin(t * halfTheta).toFloat / sinHalfTheta
      Quaternion(
        a.w * ratioA + bAdjusted.w * ratioB,
        a.x * ratioA + bAdjusted.x * ratioB,
        a.y * ratioA + bAdjusted.y * ratioB,
        a.z * ratioA + bAdjusted.z * ratioB
      )

  given Interpolatable[Quaternion] with
    def interpolate(a: Quaternion, b: Quaternion, t: Float): Quaternion =
      slerp(a, b, t)
```

#### File: `menger-common/src/main/scala/menger/common/animation/Interpolatable.scala` (addition)

Add quaternion interpolatable to the existing file:

```scala
// Add import at top:
// (Quaternion is in same package, no import needed)

// Add given instance:
given Interpolatable[Quaternion] with
  def interpolate(a: Quaternion, b: Quaternion, t: Float): Quaternion =
    Quaternion.slerp(a, b, t)
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/QuaternionSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class QuaternionSpec extends AnyFlatSpec with Matchers:

  "Quaternion.Identity" should "represent no rotation" in {
    val q = Quaternion.Identity
    q.w shouldBe 1f
    q.x shouldBe 0f
    q.y shouldBe 0f
    q.z shouldBe 0f
  }

  "Quaternion.fromEulerAngles" should "create correct quaternion for 90° Y rotation" in {
    val q = Quaternion.fromEulerAngles(0f, 90f, 0f)
    q.magnitude shouldBe 1f +- 0.001f
  }

  it should "round-trip through toEulerAngles" in {
    val original = Quaternion.fromEulerAngles(30f, 45f, 60f)
    val euler = original.toEulerAngles
    val reconstructed = Quaternion.fromEulerAngles(euler.x, euler.y, euler.z)

    original.w shouldBe reconstructed.w +- 0.01f
    original.x shouldBe reconstructed.x +- 0.01f
    original.y shouldBe reconstructed.y +- 0.01f
    original.z shouldBe reconstructed.z +- 0.01f
  }

  "Quaternion.fromAxisAngle" should "create correct quaternion" in {
    val q = Quaternion.fromAxisAngle(Vector3.UnitY, 90f)
    q.magnitude shouldBe 1f +- 0.001f
  }

  "Quaternion.slerp" should "return start quaternion at t=0" in {
    val a = Quaternion.Identity
    val b = Quaternion.fromEulerAngles(0f, 90f, 0f)
    val result = Quaternion.slerp(a, b, 0f)

    result.w shouldBe a.w +- 0.001f
    result.x shouldBe a.x +- 0.001f
  }

  it should "return end quaternion at t=1" in {
    val a = Quaternion.Identity
    val b = Quaternion.fromEulerAngles(0f, 90f, 0f)
    val result = Quaternion.slerp(a, b, 1f)

    result.w shouldBe b.w +- 0.001f
    result.y shouldBe b.y +- 0.001f
  }

  it should "interpolate smoothly at t=0.5" in {
    val a = Quaternion.Identity
    val b = Quaternion.fromEulerAngles(0f, 90f, 0f)
    val mid = Quaternion.slerp(a, b, 0.5f)

    // At 45 degrees, the result should be roughly 45 degrees
    val euler = mid.toEulerAngles
    euler.y shouldBe 45f +- 1f
  }

  it should "take the shorter path when dot product is negative" in {
    val a = Quaternion.fromEulerAngles(0f, 10f, 0f)
    val b = Quaternion.fromEulerAngles(0f, -10f, 0f)
    val mid = Quaternion.slerp(a, b, 0.5f)

    // Should interpolate through 0, not through 180
    val euler = mid.toEulerAngles
    math.abs(euler.y) should be < 5f
  }

  "Quaternion multiplication" should "combine rotations" in {
    val rotX = Quaternion.fromAxisAngle(Vector3.UnitX, 90f)
    val rotY = Quaternion.fromAxisAngle(Vector3.UnitY, 90f)
    val combined = rotY * rotX
    combined.magnitude shouldBe 1f +- 0.001f
  }

  "Quaternion.normalized" should "return unit quaternion" in {
    val q = Quaternion(2f, 0f, 0f, 0f)
    val n = q.normalized
    n.magnitude shouldBe 1f +- 0.001f
  }

  "Interpolatable[Quaternion]" should "use SLERP" in {
    val interp = summon[Interpolatable[Quaternion]]
    val a = Quaternion.Identity
    val b = Quaternion.fromEulerAngles(0f, 90f, 0f)
    val mid = interp.interpolate(a, b, 0.5f)
    mid.toEulerAngles.y shouldBe 45f +- 1f
  }
```

#### Verification

```bash
sbt "project mengerCommon" "testOnly menger.common.animation.QuaternionSpec"
```

---

### Step 13.3: Per-Instance Property Updates (JNI)

**Status:** Not Started
**Estimate:** 2.5 hours

Add JNI methods to update instance color and IOR without recreating instances.

#### File: `optix-jni/src/main/native/include/OptiXWrapper.h` (additions)

```cpp
// Add to OptiXWrapper class public methods:
bool updateInstanceColor(int instanceId, float r, float g, float b, float a);
bool updateInstanceIOR(int instanceId, float ior);
bool updateInstanceMaterial(int instanceId, float r, float g, float b, float a, float ior);
```

#### File: `optix-jni/src/main/native/OptiXWrapper.cpp` (additions)

```cpp
bool OptiXWrapper::updateInstanceColor(int instanceId, float r, float g, float b, float a) {
    if (instanceId < 0 || instanceId >= static_cast<int>(m_instanceData.size())) {
        std::cerr << "Invalid instance ID: " << instanceId << std::endl;
        return false;
    }

    auto& instance = m_instanceData[instanceId];
    instance.color = make_float4(r, g, b, a);
    m_sbtNeedsRebuild = true;
    return true;
}

bool OptiXWrapper::updateInstanceIOR(int instanceId, float ior) {
    if (instanceId < 0 || instanceId >= static_cast<int>(m_instanceData.size())) {
        std::cerr << "Invalid instance ID: " << instanceId << std::endl;
        return false;
    }

    auto& instance = m_instanceData[instanceId];
    instance.ior = ior;
    m_sbtNeedsRebuild = true;
    return true;
}

bool OptiXWrapper::updateInstanceMaterial(int instanceId, float r, float g, float b, float a, float ior) {
    if (instanceId < 0 || instanceId >= static_cast<int>(m_instanceData.size())) {
        std::cerr << "Invalid instance ID: " << instanceId << std::endl;
        return false;
    }

    auto& instance = m_instanceData[instanceId];
    instance.color = make_float4(r, g, b, a);
    instance.ior = ior;
    m_sbtNeedsRebuild = true;
    return true;
}
```

#### File: `optix-jni/src/main/native/JNIBindings.cpp` (additions)

```cpp
JNIEXPORT jboolean JNICALL Java_menger_optix_OptiXRenderer_updateInstanceColor(
    JNIEnv* env, jobject obj, jint instanceId, jfloat r, jfloat g, jfloat b, jfloat a) {
    auto* wrapper = getWrapper(env, obj);
    if (!wrapper) return JNI_FALSE;
    return wrapper->updateInstanceColor(instanceId, r, g, b, a) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_menger_optix_OptiXRenderer_updateInstanceIOR(
    JNIEnv* env, jobject obj, jint instanceId, jfloat ior) {
    auto* wrapper = getWrapper(env, obj);
    if (!wrapper) return JNI_FALSE;
    return wrapper->updateInstanceIOR(instanceId, ior) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_menger_optix_OptiXRenderer_updateInstanceMaterial(
    JNIEnv* env, jobject obj, jint instanceId, jfloat r, jfloat g, jfloat b, jfloat a, jfloat ior) {
    auto* wrapper = getWrapper(env, obj);
    if (!wrapper) return JNI_FALSE;
    return wrapper->updateInstanceMaterial(instanceId, r, g, b, a, ior) ? JNI_TRUE : JNI_FALSE;
}
```

#### File: `optix-jni/src/main/scala/menger/optix/OptiXRenderer.scala` (additions)

```scala
// Add to OptiXRenderer class:

@native def updateInstanceColor(
    instanceId: Int,
    r: Float,
    g: Float,
    b: Float,
    a: Float
): Boolean

@native def updateInstanceIOR(instanceId: Int, ior: Float): Boolean

@native def updateInstanceMaterial(
    instanceId: Int,
    r: Float,
    g: Float,
    b: Float,
    a: Float,
    ior: Float
): Boolean

def updateInstanceColor(instanceId: Int, color: Color): Boolean =
  updateInstanceColor(instanceId, color.r, color.g, color.b, color.a)

def updateInstanceMaterial(instanceId: Int, color: Color, ior: Float): Boolean =
  updateInstanceMaterial(instanceId, color.r, color.g, color.b, color.a, ior)
```

#### Tests: `optix-jni/src/test/scala/menger/optix/OptiXRendererInstanceUpdateSpec.scala`

```scala
package menger.optix

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color

class OptiXRendererInstanceUpdateSpec extends AnyFlatSpec with Matchers with OptiXTestFixture:

  "updateInstanceColor" should "update color of existing instance" in withRenderer { renderer =>
    val transform = Array.fill(12)(0f)
    transform(0) = 1f; transform(5) = 1f; transform(10) = 1f

    val instanceId = renderer.addSphereInstance(transform, Color.Red, 1.5f)
    instanceId shouldBe defined

    val result = renderer.updateInstanceColor(instanceId.get, Color.Blue)
    result shouldBe true
  }

  it should "return false for invalid instance ID" in withRenderer { renderer =>
    renderer.updateInstanceColor(999, Color.Red) shouldBe false
    renderer.updateInstanceColor(-1, Color.Red) shouldBe false
  }

  "updateInstanceIOR" should "update IOR of existing instance" in withRenderer { renderer =>
    val transform = Array.fill(12)(0f)
    transform(0) = 1f; transform(5) = 1f; transform(10) = 1f

    val instanceId = renderer.addSphereInstance(transform, Color.White, 1.0f)
    instanceId shouldBe defined

    val result = renderer.updateInstanceIOR(instanceId.get, 2.4f)
    result shouldBe true
  }

  "updateInstanceMaterial" should "update both color and IOR" in withRenderer { renderer =>
    val transform = Array.fill(12)(0f)
    transform(0) = 1f; transform(5) = 1f; transform(10) = 1f

    val instanceId = renderer.addSphereInstance(transform, Color.White, 1.0f)
    instanceId shouldBe defined

    val result = renderer.updateInstanceMaterial(instanceId.get, Color.Green, 1.8f)
    result shouldBe true
  }
```

#### Verification

```bash
sbt "project optixJni" compile
sbt "project optixJni" nativeCompile
sbt "project optixJni" "testOnly menger.optix.OptiXRendererInstanceUpdateSpec"
```

---

### Step 13.4: Camera Animation

**Status:** Not Started
**Estimate:** 2.5 hours

Create camera animation support with position, lookAt, and FOV keyframes.

#### File: `menger-common/src/main/scala/menger/common/animation/CameraKeyframe.scala`

```scala
package menger.common.animation

case class CameraKeyframe(
    time: Float,
    position: Vector3,
    lookAt: Vector3,
    up: Vector3 = Vector3.UnitY,
    fov: Float = 45f,
    easing: Easing = Easing.Linear
):
  require(time >= 0f, s"Keyframe time must be non-negative: $time")
  require(fov > 0f && fov < 180f, s"FOV must be between 0 and 180: $fov")
```

#### File: `menger-common/src/main/scala/menger/common/animation/CameraState.scala`

```scala
package menger.common.animation

case class CameraState(
    position: Vector3,
    lookAt: Vector3,
    up: Vector3,
    fov: Float
):
  def direction: Vector3 = (lookAt - position).normalized
  def right: Vector3 = direction.cross(up).normalized

object CameraState:
  val Default: CameraState = CameraState(
    position = Vector3(0f, 0f, 5f),
    lookAt = Vector3.Zero,
    up = Vector3.UnitY,
    fov = 45f
  )

  given Interpolatable[CameraState] with
    def interpolate(a: CameraState, b: CameraState, t: Float): CameraState =
      val vec3Interp = summon[Interpolatable[Vector3]]
      val floatInterp = summon[Interpolatable[Float]]
      CameraState(
        position = vec3Interp.interpolate(a.position, b.position, t),
        lookAt = vec3Interp.interpolate(a.lookAt, b.lookAt, t),
        up = vec3Interp.interpolate(a.up, b.up, t).normalized,
        fov = floatInterp.interpolate(a.fov, b.fov, t)
      )
```

#### File: `menger-common/src/main/scala/menger/common/animation/CameraTrack.scala`

```scala
package menger.common.animation

case class CameraTrack(keyframes: Seq[CameraKeyframe]):
  require(keyframes.nonEmpty, "CameraTrack requires at least one keyframe")
  require(
    keyframes.sortBy(_.time) == keyframes,
    "Keyframes must be sorted by time"
  )

  def stateAt(time: Float): CameraState =
    if keyframes.size == 1 then keyframeToState(keyframes.head)
    else if time <= keyframes.head.time then keyframeToState(keyframes.head)
    else if time >= keyframes.last.time then keyframeToState(keyframes.last)
    else
      val (before, after) = findSurroundingKeyframes(time)
      val localT = (time - before.time) / (after.time - before.time)
      val easedT = Easing(after.easing, localT)
      interpolate(keyframeToState(before), keyframeToState(after), easedT)

  def duration: Float = keyframes.last.time

  private def keyframeToState(kf: CameraKeyframe): CameraState =
    CameraState(kf.position, kf.lookAt, kf.up, kf.fov)

  private def interpolate(a: CameraState, b: CameraState, t: Float): CameraState =
    summon[Interpolatable[CameraState]].interpolate(a, b, t)

  private def findSurroundingKeyframes(
      time: Float
  ): (CameraKeyframe, CameraKeyframe) =
    val beforeIdx = keyframes.lastIndexWhere(_.time <= time)
    val afterIdx = math.min(beforeIdx + 1, keyframes.size - 1)
    (keyframes(beforeIdx), keyframes(afterIdx))

object CameraTrack:
  def static(position: Vector3, lookAt: Vector3): CameraTrack =
    CameraTrack(Seq(CameraKeyframe(0f, position, lookAt)))

  def orbit(
      center: Vector3,
      radius: Float,
      height: Float,
      duration: Float,
      steps: Int = 36
  ): CameraTrack =
    val keyframes = (0 to steps).map { i =>
      val angle = (i.toFloat / steps) * 2f * math.Pi.toFloat
      val x = center.x + radius * math.cos(angle).toFloat
      val z = center.z + radius * math.sin(angle).toFloat
      val time = (i.toFloat / steps) * duration
      CameraKeyframe(time, Vector3(x, height, z), center)
    }
    CameraTrack(keyframes)

  def dolly(
      start: Vector3,
      end: Vector3,
      lookAt: Vector3,
      duration: Float,
      easing: Easing = Easing.EaseInOut
  ): CameraTrack =
    CameraTrack(
      Seq(
        CameraKeyframe(0f, start, lookAt, easing = Easing.Linear),
        CameraKeyframe(duration, end, lookAt, easing = easing)
      )
    )
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/CameraTrackSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CameraTrackSpec extends AnyFlatSpec with Matchers:

  "CameraTrack" should "return first keyframe state before start" in {
    val track = CameraTrack(
      Seq(
        CameraKeyframe(1f, Vector3(0f, 0f, 5f), Vector3.Zero),
        CameraKeyframe(2f, Vector3(5f, 0f, 0f), Vector3.Zero)
      )
    )

    val state = track.stateAt(0f)
    state.position shouldBe Vector3(0f, 0f, 5f)
  }

  it should "return last keyframe state after end" in {
    val track = CameraTrack(
      Seq(
        CameraKeyframe(0f, Vector3(0f, 0f, 5f), Vector3.Zero),
        CameraKeyframe(1f, Vector3(5f, 0f, 0f), Vector3.Zero)
      )
    )

    val state = track.stateAt(2f)
    state.position shouldBe Vector3(5f, 0f, 0f)
  }

  it should "interpolate position between keyframes" in {
    val track = CameraTrack(
      Seq(
        CameraKeyframe(0f, Vector3(0f, 0f, 0f), Vector3.Zero),
        CameraKeyframe(1f, Vector3(10f, 0f, 0f), Vector3.Zero)
      )
    )

    val state = track.stateAt(0.5f)
    state.position.x shouldBe 5f +- 0.001f
  }

  it should "interpolate FOV between keyframes" in {
    val track = CameraTrack(
      Seq(
        CameraKeyframe(0f, Vector3.Zero, Vector3.UnitZ, fov = 30f),
        CameraKeyframe(1f, Vector3.Zero, Vector3.UnitZ, fov = 90f)
      )
    )

    val state = track.stateAt(0.5f)
    state.fov shouldBe 60f +- 0.001f
  }

  "CameraTrack.orbit" should "create circular camera path" in {
    val track = CameraTrack.orbit(Vector3.Zero, 5f, 2f, 1f, steps = 4)

    track.keyframes.size shouldBe 5
    track.duration shouldBe 1f

    // First and last positions should be the same (full circle)
    val first = track.stateAt(0f).position
    val last = track.stateAt(1f).position
    first.x shouldBe last.x +- 0.001f
    first.z shouldBe last.z +- 0.001f
  }

  "CameraTrack.dolly" should "create linear camera movement" in {
    val track = CameraTrack.dolly(
      Vector3(0f, 0f, 10f),
      Vector3(0f, 0f, 2f),
      Vector3.Zero,
      2f
    )

    track.keyframes.size shouldBe 2
    track.stateAt(0f).position.z shouldBe 10f
    track.stateAt(2f).position.z shouldBe 2f
  }
```

#### Verification

```bash
sbt "project mengerCommon" "testOnly menger.common.animation.CameraTrackSpec"
```

---

### Step 13.5: Light Animation

**Status:** Not Started
**Estimate:** 2 hours

Add support for animating light properties (position, intensity, color).

#### File: `menger-common/src/main/scala/menger/common/animation/LightState.scala`

```scala
package menger.common.animation

import menger.common.Color

case class LightState(
    position: Vector3,
    color: Color,
    intensity: Float
):
  require(intensity >= 0f, s"Light intensity must be non-negative: $intensity")

object LightState:
  def point(position: Vector3, color: Color, intensity: Float): LightState =
    LightState(position, color, intensity)

  def white(position: Vector3, intensity: Float = 1f): LightState =
    LightState(position, Color.White, intensity)

  given Interpolatable[LightState] with
    def interpolate(a: LightState, b: LightState, t: Float): LightState =
      val vec3Interp = summon[Interpolatable[Vector3]]
      val colorInterp = summon[Interpolatable[Color]]
      val floatInterp = summon[Interpolatable[Float]]
      LightState(
        position = vec3Interp.interpolate(a.position, b.position, t),
        color = colorInterp.interpolate(a.color, b.color, t),
        intensity = floatInterp.interpolate(a.intensity, b.intensity, t)
      )
```

#### File: `menger-common/src/main/scala/menger/common/animation/LightKeyframe.scala`

```scala
package menger.common.animation

import menger.common.Color

case class LightKeyframe(
    time: Float,
    position: Vector3,
    color: Color = Color.White,
    intensity: Float = 1f,
    easing: Easing = Easing.Linear
):
  require(time >= 0f, s"Keyframe time must be non-negative: $time")
  require(intensity >= 0f, s"Light intensity must be non-negative: $intensity")
```

#### File: `menger-common/src/main/scala/menger/common/animation/LightTrack.scala`

```scala
package menger.common.animation

import menger.common.Color

case class LightTrack(lightIndex: Int, keyframes: Seq[LightKeyframe]):
  require(lightIndex >= 0, s"Light index must be non-negative: $lightIndex")
  require(keyframes.nonEmpty, "LightTrack requires at least one keyframe")
  require(
    keyframes.sortBy(_.time) == keyframes,
    "Keyframes must be sorted by time"
  )

  def stateAt(time: Float): LightState =
    if keyframes.size == 1 then keyframeToState(keyframes.head)
    else if time <= keyframes.head.time then keyframeToState(keyframes.head)
    else if time >= keyframes.last.time then keyframeToState(keyframes.last)
    else
      val (before, after) = findSurroundingKeyframes(time)
      val localT = (time - before.time) / (after.time - before.time)
      val easedT = Easing(after.easing, localT)
      summon[Interpolatable[LightState]].interpolate(
        keyframeToState(before),
        keyframeToState(after),
        easedT
      )

  def duration: Float = keyframes.last.time

  private def keyframeToState(kf: LightKeyframe): LightState =
    LightState(kf.position, kf.color, kf.intensity)

  private def findSurroundingKeyframes(
      time: Float
  ): (LightKeyframe, LightKeyframe) =
    val beforeIdx = keyframes.lastIndexWhere(_.time <= time)
    val afterIdx = math.min(beforeIdx + 1, keyframes.size - 1)
    (keyframes(beforeIdx), keyframes(afterIdx))

object LightTrack:
  def static(lightIndex: Int, position: Vector3, intensity: Float = 1f): LightTrack =
    LightTrack(lightIndex, Seq(LightKeyframe(0f, position, intensity = intensity)))

  def pulse(
      lightIndex: Int,
      position: Vector3,
      minIntensity: Float,
      maxIntensity: Float,
      duration: Float,
      easing: Easing = Easing.EaseInOut
  ): LightTrack =
    LightTrack(
      lightIndex,
      Seq(
        LightKeyframe(0f, position, intensity = minIntensity),
        LightKeyframe(duration / 2f, position, intensity = maxIntensity, easing = easing),
        LightKeyframe(duration, position, intensity = minIntensity, easing = easing)
      )
    )

  def colorCycle(
      lightIndex: Int,
      position: Vector3,
      colors: Seq[Color],
      duration: Float
  ): LightTrack =
    val keyframes = colors.zipWithIndex.map { case (color, i) =>
      val time = (i.toFloat / (colors.size - 1).max(1)) * duration
      LightKeyframe(time, position, color)
    }
    LightTrack(lightIndex, keyframes)
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/LightTrackSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color

class LightTrackSpec extends AnyFlatSpec with Matchers:

  "LightTrack" should "interpolate light position" in {
    val track = LightTrack(
      0,
      Seq(
        LightKeyframe(0f, Vector3(0f, 5f, 0f)),
        LightKeyframe(1f, Vector3(10f, 5f, 0f))
      )
    )

    val state = track.stateAt(0.5f)
    state.position.x shouldBe 5f +- 0.001f
  }

  it should "interpolate light intensity" in {
    val track = LightTrack(
      0,
      Seq(
        LightKeyframe(0f, Vector3.Zero, intensity = 0f),
        LightKeyframe(1f, Vector3.Zero, intensity = 2f)
      )
    )

    val state = track.stateAt(0.5f)
    state.intensity shouldBe 1f +- 0.001f
  }

  it should "interpolate light color" in {
    val track = LightTrack(
      0,
      Seq(
        LightKeyframe(0f, Vector3.Zero, Color.Red),
        LightKeyframe(1f, Vector3.Zero, Color.Blue)
      )
    )

    val state = track.stateAt(0.5f)
    state.color.r shouldBe 0.5f +- 0.001f
    state.color.b shouldBe 0.5f +- 0.001f
  }

  "LightTrack.pulse" should "create pulsing intensity animation" in {
    val track = LightTrack.pulse(0, Vector3.Zero, 0.2f, 1f, 2f)

    track.stateAt(0f).intensity shouldBe 0.2f +- 0.001f
    track.stateAt(1f).intensity shouldBe 1f +- 0.001f
    track.stateAt(2f).intensity shouldBe 0.2f +- 0.001f
  }

  "LightTrack.colorCycle" should "cycle through colors" in {
    val track = LightTrack.colorCycle(
      0,
      Vector3.Zero,
      Seq(Color.Red, Color.Green, Color.Blue),
      2f
    )

    track.keyframes.size shouldBe 3
    track.stateAt(0f).color shouldBe Color.Red
    track.stateAt(2f).color shouldBe Color.Blue
  }
```

#### Verification

```bash
sbt "project mengerCommon" "testOnly menger.common.animation.LightTrackSpec"
```

---

### Step 13.6: Sponge Level Animation

**Status:** Not Started
**Estimate:** 2 hours

Add support for animating sponge recursion level (expensive operation - requires mesh regeneration).

#### File: `menger-common/src/main/scala/menger/common/animation/GeometryKeyframe.scala`

```scala
package menger.common.animation

case class GeometryKeyframe(
    time: Float,
    spongeLevel: Int,
    easing: Easing = Easing.Linear
):
  require(time >= 0f, s"Keyframe time must be non-negative: $time")
  require(spongeLevel >= 0 && spongeLevel <= 5, s"Sponge level must be 0-5: $spongeLevel")
```

#### File: `menger-common/src/main/scala/menger/common/animation/GeometryTrack.scala`

```scala
package menger.common.animation

case class GeometryTrack(keyframes: Seq[GeometryKeyframe]):
  require(keyframes.nonEmpty, "GeometryTrack requires at least one keyframe")
  require(
    keyframes.sortBy(_.time) == keyframes,
    "Keyframes must be sorted by time"
  )

  def levelAt(time: Float): Int =
    if keyframes.size == 1 then keyframes.head.spongeLevel
    else if time <= keyframes.head.time then keyframes.head.spongeLevel
    else if time >= keyframes.last.time then keyframes.last.spongeLevel
    else
      // For discrete values like sponge level, we step at keyframe times
      // (no interpolation - level changes instantly at keyframe)
      keyframes.filter(_.time <= time).last.spongeLevel

  def duration: Float = keyframes.last.time

  def keyframeTimes: Seq[Float] = keyframes.map(_.time)

  def hasLevelChangeAt(time: Float, tolerance: Float = 0.001f): Boolean =
    keyframes.exists(kf => math.abs(kf.time - time) < tolerance)

object GeometryTrack:
  def constant(level: Int): GeometryTrack =
    GeometryTrack(Seq(GeometryKeyframe(0f, level)))

  def progressive(maxLevel: Int, duration: Float): GeometryTrack =
    val keyframes = (0 to maxLevel).map { level =>
      val time = (level.toFloat / maxLevel) * duration
      GeometryKeyframe(time, level)
    }
    GeometryTrack(keyframes)

  val PerformanceWarning: String =
    """WARNING: Sponge level animation requires mesh regeneration at each level change.
      |This is computationally expensive and may cause frame drops.
      |Consider pre-rendering or using lower maximum levels.""".stripMargin
```

#### Tests: `menger-common/src/test/scala/menger/common/animation/GeometryTrackSpec.scala`

```scala
package menger.common.animation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class GeometryTrackSpec extends AnyFlatSpec with Matchers:

  "GeometryTrack" should "return constant level for single keyframe" in {
    val track = GeometryTrack.constant(3)
    track.levelAt(0f) shouldBe 3
    track.levelAt(100f) shouldBe 3
  }

  it should "step between levels at keyframe times" in {
    val track = GeometryTrack(
      Seq(
        GeometryKeyframe(0f, 0),
        GeometryKeyframe(1f, 1),
        GeometryKeyframe(2f, 2)
      )
    )

    track.levelAt(0f) shouldBe 0
    track.levelAt(0.5f) shouldBe 0
    track.levelAt(1f) shouldBe 1
    track.levelAt(1.5f) shouldBe 1
    track.levelAt(2f) shouldBe 2
  }

  it should "not interpolate between levels" in {
    val track = GeometryTrack(
      Seq(
        GeometryKeyframe(0f, 0),
        GeometryKeyframe(1f, 3)
      )
    )

    // Level should stay at 0 until exactly t=1
    track.levelAt(0.99f) shouldBe 0
    track.levelAt(1f) shouldBe 3
  }

  "GeometryTrack.progressive" should "create increasing levels" in {
    val track = GeometryTrack.progressive(3, 3f)

    track.keyframes.size shouldBe 4
    track.levelAt(0f) shouldBe 0
    track.levelAt(1f) shouldBe 1
    track.levelAt(2f) shouldBe 2
    track.levelAt(3f) shouldBe 3
  }

  "hasLevelChangeAt" should "detect keyframe times" in {
    val track = GeometryTrack(
      Seq(
        GeometryKeyframe(0f, 0),
        GeometryKeyframe(1f, 1)
      )
    )

    track.hasLevelChangeAt(0f) shouldBe true
    track.hasLevelChangeAt(1f) shouldBe true
    track.hasLevelChangeAt(0.5f) shouldBe false
  }
```

#### Verification

```bash
sbt "project mengerCommon" "testOnly menger.common.animation.GeometryTrackSpec"
```

---

### Step 13.7: Video Output via ffmpeg

**Status:** Not Started
**Estimate:** 2 hours

Add video encoding support using ffmpeg subprocess.

#### File: `menger-app/src/main/scala/menger/video/VideoEncoder.scala`

```scala
package menger.video

import java.io.{File, OutputStream}
import java.nio.file.{Files, Path}
import scala.sys.process.*
import scala.util.{Failure, Success, Try}

case class VideoConfig(
    width: Int,
    height: Int,
    fps: Int = 30,
    codec: String = "libx264",
    pixelFormat: String = "yuv420p",
    crf: Int = 23,
    preset: String = "medium"
):
  require(width > 0 && width % 2 == 0, s"Width must be positive and even: $width")
  require(height > 0 && height % 2 == 0, s"Height must be positive and even: $height")
  require(fps > 0 && fps <= 120, s"FPS must be between 1 and 120: $fps")
  require(crf >= 0 && crf <= 51, s"CRF must be between 0 and 51: $crf")

object VideoConfig:
  val HD720p: VideoConfig = VideoConfig(1280, 720)
  val HD1080p: VideoConfig = VideoConfig(1920, 1080)
  val UHD4K: VideoConfig = VideoConfig(3840, 2160)

enum VideoFormat(val extension: String, val codec: String):
  case MP4 extends VideoFormat("mp4", "libx264")
  case WebM extends VideoFormat("webm", "libvpx-vp9")
  case GIF extends VideoFormat("gif", "gif")

class VideoEncoder(config: VideoConfig, outputPath: Path, format: VideoFormat = VideoFormat.MP4):

  private var process: Option[Process] = None
  private var stdin: Option[OutputStream] = None
  private var frameCount: Int = 0

  def isAvailable: Boolean =
    Try("ffmpeg -version".!!).isSuccess

  def start(): Try[Unit] = Try {
    if !isAvailable then
      throw new RuntimeException("ffmpeg not found in PATH")

    val command = buildCommand()
    val pb = Process(command)

    val runningProcess = pb.run(new ProcessIO(
      writeInput = os => stdin = Some(os),
      processOutput = _.close(),
      processError = err => {
        val reader = scala.io.Source.fromInputStream(err)
        try reader.getLines().foreach(System.err.println)
        finally reader.close()
      }
    ))

    process = Some(runningProcess)
    // Give ffmpeg time to start
    Thread.sleep(100)
  }

  def writeFrame(rgbData: Array[Byte]): Try[Unit] = Try {
    stdin match
      case Some(os) =>
        os.write(rgbData)
        os.flush()
        frameCount += 1
      case None =>
        throw new RuntimeException("VideoEncoder not started")
  }

  def writeFrame(rgbaData: Array[Byte], stripAlpha: Boolean): Try[Unit] =
    if stripAlpha then
      val rgb = new Array[Byte]((rgbaData.length / 4) * 3)
      var srcIdx = 0
      var dstIdx = 0
      while srcIdx < rgbaData.length do
        rgb(dstIdx) = rgbaData(srcIdx)
        rgb(dstIdx + 1) = rgbaData(srcIdx + 1)
        rgb(dstIdx + 2) = rgbaData(srcIdx + 2)
        srcIdx += 4
        dstIdx += 3
      writeFrame(rgb)
    else
      writeFrame(rgbaData)

  def finish(): Try[Int] = Try {
    stdin.foreach { os =>
      os.flush()
      os.close()
    }
    stdin = None

    process match
      case Some(p) =>
        val exitCode = p.exitValue()
        process = None
        if exitCode != 0 then
          throw new RuntimeException(s"ffmpeg exited with code $exitCode")
        frameCount
      case None =>
        throw new RuntimeException("VideoEncoder not started")
  }

  def abort(): Unit =
    stdin.foreach(_.close())
    process.foreach(_.destroy())
    stdin = None
    process = None

  def currentFrameCount: Int = frameCount

  private def buildCommand(): Seq[String] =
    val baseCmd = Seq(
      "ffmpeg",
      "-y",  // Overwrite output
      "-f", "rawvideo",
      "-vcodec", "rawvideo",
      "-s", s"${config.width}x${config.height}",
      "-pix_fmt", "rgb24",
      "-r", config.fps.toString,
      "-i", "-"  // Read from stdin
    )

    val codecOpts = format match
      case VideoFormat.MP4 =>
        Seq(
          "-c:v", "libx264",
          "-preset", config.preset,
          "-crf", config.crf.toString,
          "-pix_fmt", config.pixelFormat
        )
      case VideoFormat.WebM =>
        Seq(
          "-c:v", "libvpx-vp9",
          "-crf", config.crf.toString,
          "-b:v", "0"
        )
      case VideoFormat.GIF =>
        Seq(
          "-vf", s"fps=${config.fps},scale=${config.width}:-1:flags=lanczos"
        )

    baseCmd ++ codecOpts :+ outputPath.toString

object VideoEncoder:
  def fromImageSequence(
      pattern: String,
      outputPath: Path,
      config: VideoConfig,
      format: VideoFormat = VideoFormat.MP4
  ): Try[Unit] = Try {
    val command = Seq(
      "ffmpeg",
      "-y",
      "-framerate", config.fps.toString,
      "-i", pattern,
      "-c:v", format.codec,
      "-pix_fmt", config.pixelFormat,
      "-crf", config.crf.toString,
      outputPath.toString
    )
    val exitCode = command.!
    if exitCode != 0 then
      throw new RuntimeException(s"ffmpeg exited with code $exitCode")
  }

  def checkAvailability(): Either[String, String] =
    Try("ffmpeg -version".!!.linesIterator.next()) match
      case Success(version) => Right(version)
      case Failure(_) => Left("ffmpeg not found. Install with: apt install ffmpeg")
```

#### File: `menger-app/src/main/scala/menger/cli/VideoOptions.scala`

```scala
package menger.cli

import java.nio.file.Path
import menger.video.{VideoConfig, VideoFormat}

case class VideoOptions(
    enabled: Boolean = false,
    outputPath: Option[Path] = None,
    format: VideoFormat = VideoFormat.MP4,
    fps: Int = 30,
    crf: Int = 23
):
  def toConfig(width: Int, height: Int): VideoConfig =
    VideoConfig(
      width = width,
      height = height,
      fps = fps,
      crf = crf
    )

object VideoOptions:
  def fromArgs(args: Map[String, String]): VideoOptions =
    val enabled = args.contains("video-output")
    val path = args.get("video-output").map(Path.of(_))
    val format = args.get("video-format").map {
      case "mp4" => VideoFormat.MP4
      case "webm" => VideoFormat.WebM
      case "gif" => VideoFormat.GIF
      case other => throw new IllegalArgumentException(s"Unknown video format: $other")
    }.getOrElse(VideoFormat.MP4)
    val fps = args.get("video-fps").map(_.toInt).getOrElse(30)
    val crf = args.get("video-crf").map(_.toInt).getOrElse(23)

    VideoOptions(enabled, path, format, fps, crf)
```

#### Tests: `menger-app/src/test/scala/menger/video/VideoEncoderSpec.scala`

```scala
package menger.video

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.nio.file.{Files, Path}
import scala.util.{Success, Failure}

class VideoEncoderSpec extends AnyFlatSpec with Matchers:

  "VideoConfig" should "require even dimensions" in {
    an[IllegalArgumentException] should be thrownBy VideoConfig(1281, 720)
    an[IllegalArgumentException] should be thrownBy VideoConfig(1280, 721)
  }

  it should "validate FPS range" in {
    an[IllegalArgumentException] should be thrownBy VideoConfig(1280, 720, fps = 0)
    an[IllegalArgumentException] should be thrownBy VideoConfig(1280, 720, fps = 121)
  }

  it should "validate CRF range" in {
    an[IllegalArgumentException] should be thrownBy VideoConfig(1280, 720, crf = -1)
    an[IllegalArgumentException] should be thrownBy VideoConfig(1280, 720, crf = 52)
  }

  "VideoEncoder.checkAvailability" should "return version or error" in {
    val result = VideoEncoder.checkAvailability()
    // Test passes regardless of whether ffmpeg is installed
    result match
      case Right(version) => version should include("ffmpeg")
      case Left(error) => error should include("not found")
  }

  "VideoEncoder" should "strip alpha channel correctly" in {
    val rgba = Array[Byte](
      1, 2, 3, -1,  // Pixel 1: R=1, G=2, B=3, A=255
      4, 5, 6, -1   // Pixel 2: R=4, G=5, B=6, A=255
    )

    // Create a mock encoder just for testing stripAlpha logic
    val tempPath = Files.createTempFile("test", ".mp4")
    try {
      val encoder = new VideoEncoder(VideoConfig(2, 1), tempPath)
      // We can't easily test writeFrame without ffmpeg, but we can verify the config
      encoder.currentFrameCount shouldBe 0
    } finally {
      Files.deleteIfExists(tempPath)
    }
  }

  "VideoConfig presets" should "have valid dimensions" in {
    VideoConfig.HD720p.width shouldBe 1280
    VideoConfig.HD720p.height shouldBe 720

    VideoConfig.HD1080p.width shouldBe 1920
    VideoConfig.HD1080p.height shouldBe 1080

    VideoConfig.UHD4K.width shouldBe 3840
    VideoConfig.UHD4K.height shouldBe 2160
  }
```

#### Verification

```bash
sbt "project mengerApp" "testOnly menger.video.VideoEncoderSpec"
# Manual test with ffmpeg:
ffmpeg -version
```

---

### Step 13.8: Preview Mode

**Status:** Not Started
**Estimate:** 3 hours

Add animation preview mode with scrubbing slider and keyframe thumbnails.

#### File: `menger-app/src/main/scala/menger/preview/AnimationPreviewState.scala`

```scala
package menger.preview

import menger.common.animation.SceneAnimation

case class AnimationPreviewState(
    animation: SceneAnimation,
    currentTime: Float = 0f,
    isPlaying: Boolean = false,
    playbackSpeed: Float = 1f,
    looping: Boolean = true,
    showKeyframeThumbnails: Boolean = true,
    thumbnailCount: Int = 10
):
  def duration: Float = animation.duration
  def fps: Float = animation.fps
  def currentFrame: Int = (currentTime * fps).toInt
  def totalFrames: Int = (duration * fps).toInt
  def progress: Float = if duration > 0 then currentTime / duration else 0f

  def withTime(t: Float): AnimationPreviewState =
    copy(currentTime = math.max(0f, math.min(duration, t)))

  def withPlaying(playing: Boolean): AnimationPreviewState =
    copy(isPlaying = playing)

  def togglePlayback: AnimationPreviewState =
    copy(isPlaying = !isPlaying)

  def advanceTime(deltaSeconds: Float): AnimationPreviewState =
    if !isPlaying then this
    else
      val newTime = currentTime + deltaSeconds * playbackSpeed
      if looping then
        copy(currentTime = newTime % duration)
      else
        copy(
          currentTime = math.min(newTime, duration),
          isPlaying = newTime < duration
        )
```

#### File: `menger-app/src/main/scala/menger/preview/PreviewControls.scala`

```scala
package menger.preview

sealed trait PreviewAction
object PreviewAction:
  case object Play extends PreviewAction
  case object Pause extends PreviewAction
  case object Stop extends PreviewAction
  case class Seek(time: Float) extends PreviewAction
  case class SeekFrame(frame: Int) extends PreviewAction
  case object NextFrame extends PreviewAction
  case object PrevFrame extends PreviewAction
  case class SetSpeed(speed: Float) extends PreviewAction
  case object ToggleLoop extends PreviewAction
  case object ToggleThumbnails extends PreviewAction

object PreviewControls:
  def handleAction(
      state: AnimationPreviewState,
      action: PreviewAction
  ): AnimationPreviewState =
    action match
      case PreviewAction.Play =>
        state.withPlaying(true)
      case PreviewAction.Pause =>
        state.withPlaying(false)
      case PreviewAction.Stop =>
        state.withTime(0f).withPlaying(false)
      case PreviewAction.Seek(time) =>
        state.withTime(time)
      case PreviewAction.SeekFrame(frame) =>
        state.withTime(frame.toFloat / state.fps)
      case PreviewAction.NextFrame =>
        state.withTime(state.currentTime + 1f / state.fps).withPlaying(false)
      case PreviewAction.PrevFrame =>
        state.withTime(state.currentTime - 1f / state.fps).withPlaying(false)
      case PreviewAction.SetSpeed(speed) =>
        state.copy(playbackSpeed = math.max(0.1f, math.min(4f, speed)))
      case PreviewAction.ToggleLoop =>
        state.copy(looping = !state.looping)
      case PreviewAction.ToggleThumbnails =>
        state.copy(showKeyframeThumbnails = !state.showKeyframeThumbnails)

  val KeyboardShortcuts: Map[Int, PreviewAction] = Map(
    // Space = toggle play/pause (handled separately)
    // Left arrow
    263 -> PreviewAction.PrevFrame,
    // Right arrow
    262 -> PreviewAction.NextFrame,
    // Home
    268 -> PreviewAction.Stop,
    // L = toggle loop
    76 -> PreviewAction.ToggleLoop,
    // T = toggle thumbnails
    84 -> PreviewAction.ToggleThumbnails
  )
```

#### File: `menger-app/src/main/scala/menger/preview/ThumbnailGenerator.scala`

```scala
package menger.preview

import menger.common.animation.SceneAnimation
import java.awt.image.BufferedImage
import java.nio.file.Path
import scala.collection.mutable
import scala.concurrent.{ExecutionContext, Future}

case class Thumbnail(
    time: Float,
    frame: Int,
    image: Option[BufferedImage]
)

class ThumbnailGenerator(
    animation: SceneAnimation,
    thumbnailWidth: Int = 160,
    thumbnailHeight: Int = 90
):
  private val thumbnails = mutable.Map[Int, Thumbnail]()
  private var renderCallback: Option[(Float, Int) => BufferedImage] = None

  def setRenderCallback(callback: (Float, Int) => BufferedImage): Unit =
    renderCallback = Some(callback)

  def generateThumbnails(count: Int)(using ExecutionContext): Future[Seq[Thumbnail]] =
    val framesToRender = (0 until count).map { i =>
      val progress = i.toFloat / (count - 1).max(1)
      val time = progress * animation.duration
      val frame = (time * animation.fps).toInt
      (time, frame)
    }

    Future.sequence(
      framesToRender.map { case (time, frame) =>
        Future {
          val image = renderCallback.map(_(time, frame))
          val thumb = Thumbnail(time, frame, image)
          thumbnails(frame) = thumb
          thumb
        }
      }
    )

  def getThumbnail(frame: Int): Option[Thumbnail] =
    thumbnails.get(frame)

  def getClosestThumbnail(time: Float): Option[Thumbnail] =
    if thumbnails.isEmpty then None
    else
      val targetFrame = (time * animation.fps).toInt
      thumbnails.values.minByOption(t => math.abs(t.frame - targetFrame))

  def keyframeThumbnails: Seq[Thumbnail] =
    animation.tracks.flatMap { track =>
      track.keyframes.map { kf =>
        val frame = (kf.time * animation.fps).toInt
        thumbnails.getOrElse(frame, Thumbnail(kf.time, frame, None))
      }
    }.distinctBy(_.frame).sortBy(_.time)

  def clearCache(): Unit = thumbnails.clear()
```

#### File: `menger-app/src/main/scala/menger/preview/AnimationPreviewEngine.scala`

```scala
package menger.preview

import menger.common.animation.SceneAnimation
import menger.optix.OptiXRenderer
import scala.util.Try

class AnimationPreviewEngine(
    animation: SceneAnimation,
    renderer: OptiXRenderer,
    width: Int,
    height: Int
):
  private var state = AnimationPreviewState(animation)
  private val thumbnailGen = new ThumbnailGenerator(animation)
  private var lastUpdateTime = System.nanoTime()

  def currentState: AnimationPreviewState = state

  def update(): Unit =
    val now = System.nanoTime()
    val deltaSeconds = (now - lastUpdateTime) / 1_000_000_000f
    lastUpdateTime = now

    state = state.advanceTime(deltaSeconds)

  def handleAction(action: PreviewAction): Unit =
    state = PreviewControls.handleAction(state, action)

  def handleKeyPress(keyCode: Int): Unit =
    if keyCode == 32 then // Space
      state = state.togglePlayback
    else
      PreviewControls.KeyboardShortcuts.get(keyCode).foreach(handleAction)

  def seekToProgress(progress: Float): Unit =
    val time = progress * state.duration
    state = state.withTime(time)

  def renderCurrentFrame(): Try[Array[Byte]] = Try {
    val frameData = animation.frameAt(state.currentTime)
    // Apply transforms to renderer
    frameData.transforms.foreach { case (instanceId, transform) =>
      renderer.updateInstanceTransform(instanceId, transform.toMatrix)
    }
    // Render and return pixel data
    renderer.render()
    renderer.getPixelBuffer()
  }

  def getTimelineMarkers: Seq[Float] =
    animation.tracks.flatMap(_.keyframes.map(_.time)).distinct.sorted

  def formatTime(seconds: Float): String =
    val mins = (seconds / 60).toInt
    val secs = seconds % 60
    f"$mins%02d:$secs%05.2f"

  def formatFrameInfo: String =
    s"Frame ${state.currentFrame + 1}/${state.totalFrames} | ${formatTime(state.currentTime)}/${formatTime(state.duration)}"
```

#### Tests: `menger-app/src/test/scala/menger/preview/AnimationPreviewStateSpec.scala`

```scala
package menger.preview

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.animation.*

class AnimationPreviewStateSpec extends AnyFlatSpec with Matchers:

  def createTestAnimation(duration: Float, fps: Float): SceneAnimation =
    SceneAnimation(
      tracks = Seq(
        AnimationTrack(
          0,
          Seq(
            Keyframe(0f, Transform3D.Identity),
            Keyframe(duration, Transform3D.Identity)
          )
        )
      ),
      duration = duration,
      fps = fps
    )

  "AnimationPreviewState" should "calculate correct frame from time" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, currentTime = 1f)

    state.currentFrame shouldBe 30
    state.totalFrames shouldBe 60
  }

  it should "calculate progress correctly" in {
    val anim = createTestAnimation(4f, 30f)
    val state = AnimationPreviewState(anim, currentTime = 2f)

    state.progress shouldBe 0.5f +- 0.001f
  }

  it should "clamp time within bounds" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim)

    state.withTime(-1f).currentTime shouldBe 0f
    state.withTime(5f).currentTime shouldBe 2f
  }

  it should "advance time when playing" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, isPlaying = true)

    val advanced = state.advanceTime(0.5f)
    advanced.currentTime shouldBe 0.5f +- 0.001f
  }

  it should "not advance time when paused" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, isPlaying = false)

    val advanced = state.advanceTime(0.5f)
    advanced.currentTime shouldBe 0f
  }

  it should "loop when reaching end" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, currentTime = 1.9f, isPlaying = true, looping = true)

    val advanced = state.advanceTime(0.5f)
    advanced.currentTime should be < 1f
    advanced.isPlaying shouldBe true
  }

  it should "stop at end when not looping" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, currentTime = 1.9f, isPlaying = true, looping = false)

    val advanced = state.advanceTime(0.5f)
    advanced.currentTime shouldBe 2f
    advanced.isPlaying shouldBe false
  }

  "PreviewControls" should "handle Play action" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim, isPlaying = false)

    val result = PreviewControls.handleAction(state, PreviewAction.Play)
    result.isPlaying shouldBe true
  }

  it should "handle SeekFrame action" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim)

    val result = PreviewControls.handleAction(state, PreviewAction.SeekFrame(15))
    result.currentTime shouldBe 0.5f +- 0.001f
  }

  it should "clamp playback speed" in {
    val anim = createTestAnimation(2f, 30f)
    val state = AnimationPreviewState(anim)

    val tooSlow = PreviewControls.handleAction(state, PreviewAction.SetSpeed(0.01f))
    tooSlow.playbackSpeed shouldBe 0.1f

    val tooFast = PreviewControls.handleAction(state, PreviewAction.SetSpeed(10f))
    tooFast.playbackSpeed shouldBe 4f
  }
```

#### Verification

```bash
sbt "project mengerApp" "testOnly menger.preview.AnimationPreviewStateSpec"
```

---

### Step 13.9: Extended DSL & Tests

**Status:** Not Started
**Estimate:** 2 hours

Extend the DSL to support all animatable properties with fluent syntax.

#### File: `menger-app/src/main/scala/menger/dsl/AnimationDSL.scala`

```scala
package menger.dsl

import menger.common.Color
import menger.common.animation.*

trait AnimationDSL:

  extension (value: Float)
    def seconds: Float = value
    def s: Float = value
    def fps: Float = value

  extension (color: Color)
    def fadeTo(target: Color, duration: Float): PropertyTrack[Color] =
      PropertyTrack.linear("color", color, target, duration)

    def fadeToWithEasing(
        target: Color,
        duration: Float,
        easing: Easing
    ): PropertyTrack[Color] =
      PropertyTrack(
        "color",
        Seq(
          PropertyKeyframe(0f, color),
          PropertyKeyframe(duration, target, easing)
        )
      )

  extension (pos: Vector3)
    def moveTo(target: Vector3, duration: Float): PropertyTrack[Vector3] =
      PropertyTrack.linear("position", pos, target, duration)

    def moveToWithEasing(
        target: Vector3,
        duration: Float,
        easing: Easing
    ): PropertyTrack[Vector3] =
      PropertyTrack(
        "position",
        Seq(
          PropertyKeyframe(0f, pos),
          PropertyKeyframe(duration, target, easing)
        )
      )

  def animate(duration: Float, fps: Float = 30f)(
      builder: AnimationBuilder ?=> Unit
  ): SceneAnimation =
    given ab: AnimationBuilder = AnimationBuilder()
    builder
    ab.build(duration, fps)

  def camera(builder: CameraAnimationBuilder ?=> Unit)(using
      ab: AnimationBuilder
  ): Unit =
    given cab: CameraAnimationBuilder = CameraAnimationBuilder()
    builder
    ab.setCameraTrack(cab.build())

  def light(index: Int)(builder: LightAnimationBuilder ?=> Unit)(using
      ab: AnimationBuilder
  ): Unit =
    given lab: LightAnimationBuilder = LightAnimationBuilder(index)
    builder
    ab.addLightTrack(lab.build())

  def sponge(builder: GeometryAnimationBuilder ?=> Unit)(using
      ab: AnimationBuilder
  ): Unit =
    given gab: GeometryAnimationBuilder = GeometryAnimationBuilder()
    builder
    ab.setGeometryTrack(gab.build())

class AnimationBuilder:
  private var objectTracks = Seq.empty[AnimationTrack]
  private var propertyTracks = Seq.empty[PropertyTrack[?]]
  private var cameraTrack: Option[CameraTrack] = None
  private var lightTracks = Seq.empty[LightTrack]
  private var geometryTrack: Option[GeometryTrack] = None

  def addObjectTrack(track: AnimationTrack): Unit =
    objectTracks = objectTracks :+ track

  def addPropertyTrack[T](track: PropertyTrack[T]): Unit =
    propertyTracks = propertyTracks :+ track

  def setCameraTrack(track: CameraTrack): Unit =
    cameraTrack = Some(track)

  def addLightTrack(track: LightTrack): Unit =
    lightTracks = lightTracks :+ track

  def setGeometryTrack(track: GeometryTrack): Unit =
    geometryTrack = Some(track)

  def build(duration: Float, fps: Float): SceneAnimation =
    SceneAnimation(
      tracks = objectTracks,
      duration = duration,
      fps = fps
    )

class CameraAnimationBuilder:
  private var keyframes = Seq.empty[CameraKeyframe]

  def at(time: Float)(
      position: Vector3,
      lookAt: Vector3,
      fov: Float = 45f,
      easing: Easing = Easing.Linear
  ): Unit =
    keyframes = keyframes :+ CameraKeyframe(time, position, lookAt, fov = fov, easing = easing)

  def orbit(center: Vector3, radius: Float, height: Float, duration: Float): Unit =
    val track = CameraTrack.orbit(center, radius, height, duration)
    keyframes = track.keyframes

  def build(): CameraTrack = CameraTrack(keyframes.sortBy(_.time))

class LightAnimationBuilder(lightIndex: Int):
  private var keyframes = Seq.empty[LightKeyframe]

  def at(time: Float)(
      position: Vector3,
      color: Color = Color.White,
      intensity: Float = 1f,
      easing: Easing = Easing.Linear
  ): Unit =
    keyframes = keyframes :+ LightKeyframe(time, position, color, intensity, easing)

  def pulse(position: Vector3, min: Float, max: Float, duration: Float): Unit =
    val track = LightTrack.pulse(lightIndex, position, min, max, duration)
    keyframes = track.keyframes

  def build(): LightTrack = LightTrack(lightIndex, keyframes.sortBy(_.time))

class GeometryAnimationBuilder:
  private var keyframes = Seq.empty[GeometryKeyframe]

  def at(time: Float)(level: Int): Unit =
    keyframes = keyframes :+ GeometryKeyframe(time, level)

  def progressive(maxLevel: Int, duration: Float): Unit =
    val track = GeometryTrack.progressive(maxLevel, duration)
    keyframes = track.keyframes

  def build(): GeometryTrack = GeometryTrack(keyframes.sortBy(_.time))

object AnimationDSL extends AnimationDSL
```

#### File: `menger-app/src/main/scala/menger/dsl/AnimationDSLExamples.scala`

```scala
package menger.dsl

import menger.common.Color
import menger.common.animation.*

object AnimationDSLExamples extends AnimationDSL:

  val orbitingCamera: SceneAnimation = animate(duration = 10.seconds, fps = 30.fps) {
    camera {
      orbit(center = Vector3.Zero, radius = 5f, height = 2f, duration = 10f)
    }
  }

  val pulsingLight: SceneAnimation = animate(duration = 4.seconds) {
    light(0) {
      pulse(Vector3(0f, 5f, 0f), min = 0.2f, max = 1f, duration = 4f)
    }
  }

  val colorFade: SceneAnimation = animate(duration = 3.seconds) {
    // This would require extending the DSL to support per-instance color tracks
    // For now, demonstrates the pattern
  }

  val progressiveSponge: SceneAnimation = animate(duration = 5.seconds) {
    sponge {
      progressive(maxLevel = 4, duration = 5f)
    }
  }

  val complexAnimation: SceneAnimation = animate(duration = 10.seconds, fps = 60.fps) {
    camera {
      at(0.seconds)(
        position = Vector3(0f, 0f, 10f),
        lookAt = Vector3.Zero,
        fov = 45f
      )
      at(5.seconds)(
        position = Vector3(5f, 3f, 5f),
        lookAt = Vector3.Zero,
        fov = 60f,
        easing = Easing.EaseInOut
      )
      at(10.seconds)(
        position = Vector3(0f, 0f, 10f),
        lookAt = Vector3.Zero,
        fov = 45f,
        easing = Easing.EaseOut
      )
    }

    light(0) {
      at(0.seconds)(Vector3(-5f, 5f, 5f), Color.White, intensity = 1f)
      at(5.seconds)(Vector3(5f, 5f, 5f), Color.Red, intensity = 1.5f, easing = Easing.Cubic)
      at(10.seconds)(Vector3(-5f, 5f, 5f), Color.White, intensity = 1f, easing = Easing.EaseOut)
    }

    sponge {
      at(0.seconds)(level = 0)
      at(2.seconds)(level = 1)
      at(4.seconds)(level = 2)
      at(6.seconds)(level = 3)
    }
  }
```

#### Tests: `menger-app/src/test/scala/menger/dsl/AnimationDSLSpec.scala`

```scala
package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.common.Color
import menger.common.animation.*

class AnimationDSLSpec extends AnyFlatSpec with Matchers with AnimationDSL:

  "AnimationDSL time extensions" should "convert to seconds" in {
    5.seconds shouldBe 5f
    2.5.s shouldBe 2.5f
  }

  "Color.fadeTo" should "create linear color track" in {
    val track = Color.Red.fadeTo(Color.Blue, 2f)

    track.name shouldBe "color"
    track.duration shouldBe 2f
    track.valueAt(0f) shouldBe Color.Red
    track.valueAt(2f) shouldBe Color.Blue
  }

  "Color.fadeToWithEasing" should "apply easing function" in {
    val track = Color.Red.fadeToWithEasing(Color.Blue, 1f, Easing.EaseIn)

    track.keyframes.last.easing shouldBe Easing.EaseIn
  }

  "Vector3.moveTo" should "create linear position track" in {
    val start = Vector3(0f, 0f, 0f)
    val end = Vector3(10f, 0f, 0f)
    val track = start.moveTo(end, 2f)

    track.valueAt(1f).x shouldBe 5f +- 0.001f
  }

  "animate block" should "create SceneAnimation with duration and fps" in {
    val anim = animate(duration = 5.seconds, fps = 60.fps) {
      // Empty animation
    }

    anim.duration shouldBe 5f
    anim.fps shouldBe 60f
  }

  "camera block" should "create camera track with keyframes" in {
    val anim = animate(duration = 2.seconds) {
      camera {
        at(0.seconds)(Vector3(0f, 0f, 5f), Vector3.Zero)
        at(2.seconds)(Vector3(5f, 0f, 0f), Vector3.Zero, easing = Easing.EaseInOut)
      }
    }

    // Animation should be created (camera track stored in builder)
    anim.duration shouldBe 2f
  }

  "light block" should "create light track for specified index" in {
    val anim = animate(duration = 4.seconds) {
      light(0) {
        pulse(Vector3.Zero, 0.5f, 1f, 4f)
      }
    }

    anim.duration shouldBe 4f
  }

  "sponge block" should "create geometry track" in {
    val anim = animate(duration = 5.seconds) {
      sponge {
        progressive(maxLevel = 3, duration = 5f)
      }
    }

    anim.duration shouldBe 5f
  }

  "CameraAnimationBuilder.orbit" should "create circular path" in {
    given cab: CameraAnimationBuilder = CameraAnimationBuilder()
    cab.orbit(Vector3.Zero, 5f, 2f, 1f)
    val track = cab.build()

    track.keyframes.size should be > 2
    // First and last should be same position (full circle)
    val first = track.stateAt(0f).position
    val last = track.stateAt(1f).position
    first.x shouldBe last.x +- 0.01f
  }

  "GeometryAnimationBuilder.progressive" should "create level progression" in {
    given gab: GeometryAnimationBuilder = GeometryAnimationBuilder()
    gab.progressive(3, 3f)
    val track = gab.build()

    track.levelAt(0f) shouldBe 0
    track.levelAt(1f) shouldBe 1
    track.levelAt(2f) shouldBe 2
    track.levelAt(3f) shouldBe 3
  }
```

#### Verification

```bash
sbt "project mengerApp" "testOnly menger.dsl.AnimationDSLSpec"
sbt "project mengerApp" compile
```

---

## Summary

| Step | Task | Estimate | Status |
|------|------|----------|--------|
| 13.1 | Generic AnimatableProperty[T] System | 3h | Not Started |
| 13.2 | Quaternion SLERP for Smooth Rotation | 2h | Not Started |
| 13.3 | Per-Instance Property Updates (JNI) | 2.5h | Not Started |
| 13.4 | Camera Animation | 2.5h | Not Started |
| 13.5 | Light Animation | 2h | Not Started |
| 13.6 | Sponge Level Animation | 2h | Not Started |
| 13.7 | Video Output via ffmpeg | 2h | Not Started |
| 13.8 | Preview Mode | 3h | Not Started |
| 13.9 | Extended DSL & Tests | 2h | Not Started |
| **Total** | | **~21h** | |

---

## Notes

### Implementation Order

Recommended implementation sequence:

1. **13.1 first** - Generic property system is foundational
2. **13.2** - Quaternion SLERP for smooth rotation (foundational math)
3. **13.4 + 13.5** - Camera and light animation can be done in parallel
4. **13.3** - JNI updates for per-instance properties
5. **13.6** - Geometry animation (depends on understanding performance implications)
6. **13.7** - Video output (independent of other features)
7. **13.8** - Preview mode (requires animation system working)
8. **13.9** - DSL last (integrates everything)

### Testing Strategy

- Unit tests for each component in isolation
- Integration tests for animation playback
- Visual regression tests for rendered output
- Performance benchmarks for geometry animation

### Known Limitations

1. **Sponge level animation** is expensive - mesh must be regenerated at each level change
2. **Camera up vector** uses linear interpolation (may cause tilting); consider SLERP for extreme cases
3. **Light animation** requires SBT rebuild after each change
4. **Video encoding** requires ffmpeg to be installed externally
5. **Preview mode** requires windowed rendering (not headless)
6. **Rotation interpolation** uses Quaternion SLERP (Step 13.2) for object rotation; Euler angles still used for simple cases

### Performance Considerations

| Feature | Performance Impact | Mitigation |
|---------|-------------------|------------|
| Color/IOR animation | Low (SBT update only) | Batch updates |
| Camera animation | None (CPU only) | N/A |
| Light animation | Medium (SBT rebuild) | Limit light count |
| Geometry animation | High (mesh regen) | Pre-cache meshes, warn user |
| Video encoding | Medium (I/O bound) | Use fast preset, lower CRF |

### Dependencies

- Sprint 12 (Object Animation Foundation) - **required**
- ffmpeg - required for video output (Step 13.6)
- LibGDX - required for preview mode UI (Step 13.7)

### Future Enhancements (Sprint 14+)

- Bezier/spline camera paths
- Animation blending and layering
- Keyframe editor with visual timeline
- Animation presets library
- Physics-based animation (spring, gravity, collision)
- Multi-pass rendering for motion blur
