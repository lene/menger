package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera

class MengerInputMultiplexer(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends InputMultiplexer:
  addProcessor(CameraController(camera, eventDispatcher))
  addProcessor(KeyController(camera, eventDispatcher))
