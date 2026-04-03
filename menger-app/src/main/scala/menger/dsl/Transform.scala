package menger.dsl

/** Local-space transform: translation, rotation (Euler angles in radians), and uniform scale.
  *
  * When composing a hierarchy, child world transform = accumulate(parent_world, child_local).
  * Translation accumulation scales child translation by parent scale (standard scene graph behaviour).
  * Rotation accumulation adds Euler angles (approximate; order-dependent for large rotations).
  * Scale accumulation multiplies.
  */
case class Transform(
  translation: Vec3 = Vec3.Zero,
  rotation: Vec3 = Vec3.Zero,
  scale: Float = 1.0f
)

object Transform:
  val Identity: Transform = Transform()

  def at(pos: Vec3): Transform = Transform(translation = pos)

  def scaled(s: Float): Transform = Transform(scale = s)

  def accumulate(parent: Transform, child: Transform): Transform =
    Transform(
      translation = parent.translation + (child.translation * parent.scale),
      rotation    = parent.rotation    + child.rotation,
      scale       = parent.scale       * child.scale
    )
