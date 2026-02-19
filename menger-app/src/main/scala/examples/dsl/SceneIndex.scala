package examples.dsl

/** Forces initialization of all example scene objects so their SceneRegistry.register() calls
  * execute. Referenced from Main before scene loading to enable short-name lookups. */
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
  )
