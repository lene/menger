package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
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
      Right(())

  override def buildScene(specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int): Try[Unit] = Try:
    specs.foreach { spec =>
      val cd = spec.curveData.get
      val material = MaterialExtractor.extract(spec)
      requireInstanceId(
        renderer.addCurveInstance(cd.points.toArray, cd.widths.toArray, material),
        s"curve instance (${cd.points.length / 3} control points)"
      )
    }

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean =
    spec1.objectType == "curve" && spec2.objectType == "curve"

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
