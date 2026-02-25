package menger.dsl

/** Test fixture for SceneLoaderSuite: an animated scene parameterized by t. */
object TestAnimatedSceneObject:
  def scene(t: Float): Scene =
    val x = 2f * math.cos(t).toFloat
    val z = 2f * math.sin(t).toFloat
    Scene(
      camera = Camera.Default,
      objects = List(Sphere(pos = Vec3(x, 0f, z), material = Material.Chrome, size = 0.5f))
    )
