package menger.dsl

/** A node in the scene graph tree.
  *
  * Each node carries a local-space transform, an optional material override that is inherited by
  * all descendant nodes (unless a descendant sets its own material), optional leaf geometry, and
  * an optional list of child nodes.
  *
  * Material resolution priority (highest to lowest):
  *   1. Geometry's own material (set directly on the SceneObject)
  *   2. Nearest ancestor SceneNode.material
  *   3. No material (rendering default applies)
  *
  * Transform accumulation: a child's world transform =
  *   Transform.accumulate(parent_world_transform, child.transform)
  */
case class SceneNode(
  transform: Transform = Transform.Identity,
  material: Option[Material] = None,
  geometry: Option[SceneObject] = None,
  children: List[SceneNode] = Nil
):
  def allLeafGeometry: List[SceneObject] =
    geometry.toList ++ children.flatMap(_.allLeafGeometry)

object SceneNode:
  def leaf(obj: SceneObject): SceneNode =
    SceneNode(geometry = Some(obj))

  def leaf(transform: Transform, obj: SceneObject): SceneNode =
    SceneNode(transform = transform, geometry = Some(obj))

  def group(children: SceneNode*): SceneNode =
    SceneNode(children = children.toList)

  def group(transform: Transform, children: SceneNode*): SceneNode =
    SceneNode(transform = transform, children = children.toList)

  def group(material: Material, children: SceneNode*): SceneNode =
    SceneNode(material = Some(material), children = children.toList)
