// One class per governed layer that depends on something its layer may not (menger#90 (e)).
// They live in test sources, which the architecture rules never import (DoNotIncludeTests);
// ArchitectureRulesCanFailSpec imports them on purpose and shows each layer's rule failing.

package menger.objects.planted:
  object PlantedObjects:
    def leak(scene: menger.dsl.Scene): Unit = ()

package menger.dsl.planted:
  object PlantedDsl:
    def leak(engine: menger.engines.RenderEngine): Unit = ()

package menger.config.planted:
  object PlantedConfig:
    def leak(engine: menger.engines.RenderEngine): Unit = ()

package menger.input.planted:
  object PlantedInput:
    def leak(scene: menger.dsl.Scene): Unit = ()

package menger.cli.planted:
  object PlantedCli:
    def leak(engine: menger.engines.RenderEngine): Unit = ()

package menger.spec.planted:
  object PlantedSpec:
    def leak(scene: menger.dsl.Scene): Unit = ()

package menger.scene.planted:
  object PlantedScene:
    def leak(engine: menger.engines.RenderEngine): Unit = ()

package menger.video.planted:
  object PlantedVideo:
    def leak(color: menger.common.Color): Unit = ()
