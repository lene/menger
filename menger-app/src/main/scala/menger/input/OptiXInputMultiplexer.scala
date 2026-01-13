package menger.input

import com.badlogic.gdx.InputMultiplexer

class OptiXInputMultiplexer(
  cameraController: OptiXCameraController,
  dispatcher: EventDispatcher
) extends InputMultiplexer:
  addProcessor(cameraController)
  addProcessor(OptiXKeyController(dispatcher))
