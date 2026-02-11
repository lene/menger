package menger.dsl

/** Test scene object for SceneLoaderSuite reflection testing. */
object TestSceneObject:
  val scene = Scene(
    camera = Camera.Default,
    objects = List(Sphere(Material.Glass))
  )
