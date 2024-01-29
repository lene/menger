package menger

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController

class MengerInputMultiplexer(camera: PerspectiveCamera) extends InputMultiplexer:
  addProcessor(new CameraInputController(camera))
  addProcessor(new InputController(camera))
