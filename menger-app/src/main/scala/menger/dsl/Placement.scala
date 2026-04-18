package menger.dsl

import scala.util.Random

object Placement:

  /** Grid of copies in the XZ plane, centered at the origin.
   *  @param rows    number of rows (Z axis)
   *  @param cols    number of columns (X axis)
   *  @param spacing distance between adjacent copies
   *  @param node    prototype node to replicate
   */
  def grid(rows: Int, cols: Int, spacing: Float)(node: SceneNode): SceneNode =
    require(rows > 0 && cols > 0, "grid requires rows > 0 and cols > 0")
    val offsetX = (cols - 1) * spacing * 0.5f
    val offsetZ = (rows - 1) * spacing * 0.5f
    val children = for
      r <- 0 until rows
      c <- 0 until cols
    yield
      val pos = Vec3(c * spacing - offsetX, 0f, r * spacing - offsetZ)
      SceneNode.group(Transform.at(pos), node)
    SceneNode.group(children.toList*)

  /** Ring of evenly-spaced copies around the Y axis.
   *  @param count  number of copies
   *  @param radius distance from origin
   *  @param node   prototype node to replicate
   */
  def ring(count: Int, radius: Float)(node: SceneNode): SceneNode =
    require(count > 0, "ring requires count > 0")
    val step = 2f * math.Pi.toFloat / count
    val children = (0 until count).map { i =>
      val angle = i * step
      val pos   = Vec3(radius * math.cos(angle).toFloat, 0f, radius * math.sin(angle).toFloat)
      SceneNode.group(Transform.at(pos), node)
    }
    SceneNode.group(children.toList*)

  /** Archimedes spiral in the XZ plane.
   *  Radius interpolates linearly from radiusStart to radiusEnd over `turns` full revolutions.
   *  @param count       number of copies
   *  @param radiusStart radius at the first copy
   *  @param radiusEnd   radius at the last copy
   *  @param turns       number of full revolutions over all copies
   *  @param node        prototype node to replicate
   */
  def spiral(
    count: Int,
    radiusStart: Float,
    radiusEnd: Float,
    turns: Float
  )(node: SceneNode): SceneNode =
    require(count > 0, "spiral requires count > 0")
    val children = (0 until count).map { i =>
      val frac   = if count > 1 then i.toFloat / (count - 1) else 0f
      val angle  = frac * turns * 2f * math.Pi.toFloat
      val radius = radiusStart + frac * (radiusEnd - radiusStart)
      val pos    = Vec3(radius * math.cos(angle).toFloat, 0f, radius * math.sin(angle).toFloat)
      SceneNode.group(Transform.at(pos), node)
    }
    SceneNode.group(children.toList*)

  /** Random scatter within a centred axis-aligned box.
   *  @param count  number of copies
   *  @param bounds half-extents: copies placed in [-bounds.x, bounds.x] × [-bounds.y, bounds.y] × [-bounds.z, bounds.z]
   *  @param seed   RNG seed for reproducibility (default 42)
   *  @param node   prototype node to replicate
   */
  def scatter(count: Int, bounds: Vec3, seed: Long = 42L)(node: SceneNode): SceneNode =
    require(count > 0, "scatter requires count > 0")
    val rng = Random(seed)
    def rand(extent: Float): Float = (rng.nextFloat() * 2f - 1f) * extent
    val children = (0 until count).map { _ =>
      val pos = Vec3(rand(bounds.x), rand(bounds.y), rand(bounds.z))
      SceneNode.group(Transform.at(pos), node)
    }
    SceneNode.group(children.toList*)
