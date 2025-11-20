package menger.common

case class ImageSize(width: Int, height: Int):
  require(width > 0, s"width must be positive, got $width")
  require(height > 0, s"height must be positive, got $height")
