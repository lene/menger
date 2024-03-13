package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import menger.input.MengerInputController

class MengerInputMultiplexer(camera: PerspectiveCamera) extends InputMultiplexer:
  addProcessor(new MengerCameraInputController(camera))
  addProcessor(new MengerInputController(camera))
