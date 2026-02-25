package examples.dsl

/** Forces initialization of all example scene objects so their SceneRegistry.register() calls
  * execute. Referenced from Main before scene loading to enable short-name lookups.
  *
  * Animated scenes (def scene(t: Float)) are listed separately -- they don't register short
  * names but are referenced here for completeness and class loading.
  */
object SceneIndex:
  val all: List[menger.dsl.Scene] = List(
    SimpleScene.scene,
    ThreeMaterials.scene,
    GlassSphere.scene,
    FilmSphere.scene,
    MengerShowcase.scene,
    SpongeShowcase.scene,
    CausticsDemo.scene,
    CustomMaterials.scene,
    ComplexLighting.scene,
    ReusableComponents.scene,
    TesseractDemo.scene,
  )

  /** Animated scene objects (loaded via reflection with def scene(t: Float)). */
  val animated: List[Float => menger.dsl.Scene] = List(
    OrbitingSphere.scene,
    PulsingSponge.scene,
  )
