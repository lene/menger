package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.objects.LSystemGrammar

class LSystemSceneBuilder(textureDir: String = ".") extends SceneBuilder:

  override def validate(specs: List[ObjectSpec], maxInstances: Int): Either[String, Unit] =
    if specs.isEmpty then Left("Object specs list cannot be empty")
    else if !specs.forall(_.objectType == "lsystem") then
      Left("All objects must be lsystem for LSystemSceneBuilder")
    else Right(())

  override def buildScene(
    specs: List[ObjectSpec], renderer: OptiXRenderer, maxInstances: Int
  ): Try[Unit] = Try:
    specs.foreach { spec =>
      val generatedSpecs = generateFromSpec(spec)
      val grouped = generatedSpecs.groupBy(_.objectType)
      grouped.foreach { case (objType, objSpecs) =>
        val builder = resolveSubBuilder(objType)
        builder.buildScene(objSpecs, renderer, maxInstances).get
      }
    }

  private def generateFromSpec(spec: ObjectSpec): List[ObjectSpec] =
    val grammar = LSystemGrammar("F", Map('F' -> Seq((1.0, "F[+F]F[-F]F"))))
    val rewritten = grammar.rewrite(4)
    val turtle = LSystemTurtle3D(rewritten, 25.7f, 0.1f, 0.05f)
    turtle.generate()

  private def resolveSubBuilder(objType: String): SceneBuilder = objType match
    case "curve" => CurveSceneBuilder(textureDir)
    case "sphere" => SphereSceneBuilder(textureDir)
    case "cone" => ConeSceneBuilder(textureDir)
    case _ => CurveSceneBuilder(textureDir)

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean = true

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
