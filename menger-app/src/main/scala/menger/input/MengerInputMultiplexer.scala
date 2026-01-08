package menger.input

import com.badlogic.gdx.InputMultiplexer
import com.badlogic.gdx.graphics.PerspectiveCamera

class MengerInputMultiplexer(
  camera: PerspectiveCamera, eventDispatcher: EventDispatcher
) extends InputMultiplexer:
  addProcessor(GdxCameraController(camera, eventDispatcher))
  addProcessor(GdxKeyController(camera, eventDispatcher))
