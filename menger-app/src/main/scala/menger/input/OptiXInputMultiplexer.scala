package menger.input

import com.badlogic.gdx.InputMultiplexer

class OptiXInputMultiplexer(
  cameraController: OptiXCameraController,
  keyController: OptiXKeyController
) extends InputMultiplexer:
  addProcessor(cameraController)
  addProcessor(keyController)
