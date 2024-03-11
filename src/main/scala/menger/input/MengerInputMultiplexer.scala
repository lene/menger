package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController
import menger.input.InputController3D

class MengerInputMultiplexer(camera: PerspectiveCamera) extends InputMultiplexer:
  addProcessor(new CameraInputController(camera))
  addProcessor(new InputController3D(camera))
