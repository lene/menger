package menger.objects


enum Direction:
  case X, Y, Z, _X, _Y, _Z
  
case class Face(xCen: Float, yCen: Float, zCen: Float, scale: Float, normal: Direction):
  def subdivide(): List[Face] = ???
  