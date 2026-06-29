package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectRotation
import menger.ObjectSpec

class CurveSceneBuilder(textureDir: String = ".") extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then
      Left("Object specs list cannot be empty")
    else if specs.length > maxInstances then
      Left(s"Too many curve objects: ${specs.length} exceeds max instances limit of $maxInstances")
    else if !specs.forall(_.objectType == "curve") then
      Left("All objects must be curves for CurveSceneBuilder")
    else if specs.exists(_.curveData.isEmpty) then
      Left("All curve specs must have curveData populated")
    else
      val unsupported = specs.collect {
        case s if s.texture.isDefined => "texture"
        case s if s.videoTexture.isDefined => "videoTexture"
        case s if s.normalMap.isDefined => "normalMap"
        case s if s.roughnessMap.isDefined => "roughnessMap"
        case s if s.proceduralType != 0 => "proceduralType"
        case s if s.rotation != ObjectRotation() => "rotation"
        case s if s.projection4D.isDefined => "projection4D"
        case s if s.level.isDefined => "level"
        case s if s.edgeRadius.isDefined => "edgeRadius"
        case s if s.edgeMaterial.isDefined => "edgeMaterial"
      }
      if unsupported.nonEmpty then
        Left(s"Curve objects do not support: ${unsupported.distinct.mkString(", ")}. " +
          "Curves accept only color, ior, material (roughness/metallic/specular/emission), " +
          "and curveData (control points, widths, closed).")
      else Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    specs.foreach { spec =>
      val cd = spec.curveData.get
      val material = MaterialExtractor.extract(spec)
      val (paddedPoints, paddedWidths) = padToMinPoints(cd.points, cd.widths, 4)
      requireInstanceId(
        renderer.addCurveInstance(paddedPoints.toArray, paddedWidths.toArray, material),
        s"curve instance (${paddedPoints.length / 3} control points)"
      )
    }

  private def padToMinPoints(
    points: Vector[Float], widths: Vector[Float], minPoints: Int
  ): (Vector[Float], Vector[Float]) =
    val numPoints = points.length / 3
    if numPoints >= minPoints then (points, widths)
    else
      val repeats = minPoints - numPoints
      val lastX = points(points.length - 3)
      val lastY = points(points.length - 2)
      val lastZ = points(points.length - 1)
      val lastW = widths.last
      val paddedPoints = points ++ Vector.fill(repeats * 3)(0f)
        .zipWithIndex.map { case (_, i) =>
          i % 3 match
            case 0 => lastX
            case 1 => lastY
            case 2 => lastZ
        }
      val paddedWidths = widths ++ Vector.fill(repeats)(lastW)
      (paddedPoints, paddedWidths)

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType == "curve" && spec2.objectType == "curve"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
