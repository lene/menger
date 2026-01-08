package menger.input

import com.badlogic.gdx.InputMultiplexer

class OptiXInputMultiplexer(cameraController: OptiXCameraController) extends InputMultiplexer:
  addProcessor(cameraController)
  addProcessor(OptiXKeyController())
