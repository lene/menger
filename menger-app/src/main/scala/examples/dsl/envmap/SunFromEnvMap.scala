package examples.dsl.envmap

import scala.language.implicitConversions

import menger.dsl._

/**
 * Example: the sun of an environment map as the scene's key light (`Sun.fromEnvMap`).
 *
 * The map's sun is a few very bright texels. `Sun.fromEnvMap` measures where it is, its colour and
 * how strongly it lights a matte surface, and makes a `Sun` of it: the shadow gets a penumbra and,
 * as the first light, it emits the caustic photons. With the map as `envMap` and `ibl` on, the
 * renderer leaves the map's own sun out of the image-based lighting, so the sun lights the scene
 * once, as this light, and the rest of the sky as the environment. Image-based lighting is
 * physical: a matte surface returns its albedo times the map's irradiance over pi, there is no
 * ambient fill, and `ACES` tone mapping brings the sun's strength into range.
 *
 * In a sub-package, so it is not in `SceneIndex` (building the scene reads the map) and not in
 * the agent's corpus. It is loaded by name, from where the map is:
 * --scene examples.dsl.envmap.SunFromEnvMap --texture-dir menger-app/src/test/resources
 */
object SunFromEnvMap:
  private val envMapPath = "rogland_sunset_2k.hdr"

  val scene: Scene = Scene(
    camera = Camera(
      position = (0f, 2.6f, 4.2f),
      lookAt = (0f, -0.2f, 0f)
    ),
    objects = List(
      Cylinder(
        p0 = Some(Vec3(0f, -0.65f, 0f)),
        p1 = Some(Vec3(0f, -0.45f, 0f)),
        radius = Some(1.8f),
        material = Some(Material.matte(Color(0.5f, 0.48f, 0.44f)))
      ),
      // 0.01 above the plinth's top cap, as in SunKeyLight.
      Cube(pos = (0f, -0.04f, 0f), material = Some(Material.Glass), size = 0.8f)
    ),
    lights = List(Sun.fromEnvMap(envMapPath)),
    envMap = Some(envMapPath),
    ibl = Some(IBL(samples = 2)),
    toneMapping = ToneMapping.ACES(exposure = 1.0f),
    caustics = Some(Caustics(photonsPerIteration = 200000, iterations = 10)),
    render = Some(RenderSettings(shadows = true, accumulation = 4))
  )
