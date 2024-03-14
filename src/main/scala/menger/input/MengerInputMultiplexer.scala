package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import menger.input.MengerKeyInputController

class MengerInputMultiplexer(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends InputMultiplexer:
  addProcessor(new MengerCameraInputController(camera, eventDispatcher))
  addProcessor(new MengerKeyInputController(camera, eventDispatcher))
