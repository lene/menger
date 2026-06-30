package menger.engines.scene

import scala.util.Try

import io.github.lene.optix.OptiXRenderer
import menger.ObjectSpec
import menger.objects.LSystemGrammar
import menger.objects.LSystemPresets

/** Sealed sub-builder type — compile-time safety replacing raw String dispatch (A4, Sprint 32). */
enum SubBuilderType:
  case Curve, Sphere, Cone

  def resolve(textureDir: String): SceneBuilder =
    this match
      case Curve  => CurveSceneBuilder(textureDir)
      case Sphere => SphereSceneBuilder(textureDir)
      case Cone   => ConeSceneBuilder(textureDir)

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
        val subType = objType.toLowerCase match
          case "curve"  => SubBuilderType.Curve
          case "sphere" => SubBuilderType.Sphere
          case "cone"   => SubBuilderType.Cone
          case _        => SubBuilderType.Curve
        val builder = resolveSubBuilder(subType)
        builder.buildScene(objSpecs, renderer, maxInstances).get
      }
    }

  private def generateFromSpec(spec: ObjectSpec): List[ObjectSpec] =
    val preset = spec.lsystemPreset.getOrElse("tree")
    val level = spec.level.map(_.toInt).getOrElse(4)
    val size = spec.size
    val angle = spec.lsystemAngle.getOrElse(25.7f)
    val seed = spec.lsystemSeed.getOrElse(42L)

    val (axiom, rules, defAngle, segLen, initWidth, decay, _) =
      LSystemPresets(preset)
    val grammar = LSystemGrammar(axiom, rules.view.mapValues(v => Seq((1.0, v))).toMap, seed)
    val rewritten = grammar.rewrite(level)

    if spec.lsystemDim == 4 then
      val proj4D = spec.projection4D.getOrElse(menger.Projection4DSpec.default)
      val turtle = LSystemTurtle4D(
        rewritten,
        if spec.lsystemAngle.isDefined then angle else defAngle,
        segLen * size,
        initWidth * size,
        decay,
        seed,
        rotXW = proj4D.rotXW,
        rotYW = proj4D.rotYW,
        rotZW = proj4D.rotZW,
        eyeW = proj4D.eyeW,
        screenW = proj4D.screenW
      )
      turtle.generate()
    else
      val turtle = LSystemTurtle3D(
        rewritten,
        if spec.lsystemAngle.isDefined then angle else defAngle,
        segLen * size,
        initWidth * size,
        decay
      )
      turtle.generate()

  private def resolveSubBuilder(builderType: SubBuilderType): SceneBuilder =
    builderType.resolve(textureDir)

  override def isCompatible(spec1: ObjectSpec, spec2: ObjectSpec): Boolean = true

  override def calculateInstanceCount(specs: List[ObjectSpec]): Long =
    specs.length.toLong
