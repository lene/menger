package menger.config

case class LevelConfig(warnLevel: Int, maxLevel: Int, estimateTriangles: Int => Long)
