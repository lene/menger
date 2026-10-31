package menger.dsl

import scala.compiletime.constValueTuple
import scala.compiletime.erasedValue
import scala.deriving.Mirror

import menger.spec.Degrees
import menger.spec.Projection4DSpec
import menger.spec.Radians

/** The unit of an angle-valued field, for the manifest's `unit` (menger#95).
  *
  * It is read off the case class's own field types, so a field retyped from `Degrees` to
  * `Radians` changes its unit in the manifest without anyone editing a table. The one hand-kept
  * entry is the 3D `rotation: Vec3` of every scene object: a `Vec3` carries no unit, so the
  * name says it (radians). */
object FieldUnits:

  val Degrees = "degrees"
  val Radians = "radians"

  private inline def unitOf[T]: Option[String] =
    inline erasedValue[T] match
      case _: menger.spec.Degrees => Some(Degrees)
      case _: Option[menger.spec.Degrees] => Some(Degrees)
      case _: menger.spec.Radians => Some(Radians)
      case _ => None

  private inline def unitsOfElems[Elems <: Tuple]: List[Option[String]] =
    inline erasedValue[Elems] match
      case _: EmptyTuple => Nil
      case _: (h *: t) => unitOf[h] :: unitsOfElems[t]

  /** Field name -> unit, for the angle-valued fields of the case class `T`. */
  inline def derive[T](using m: Mirror.ProductOf[T]): Map[String, String] =
    val labels = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
    labels.zip(unitsOfElems[m.MirroredElemTypes]).collect { case (label, Some(unit)) =>
      label -> unit
    }.toMap

  private val byType: Map[String, Map[String, String]] = Map(
    "Projection4DSpec" -> derive[Projection4DSpec],
    "LSystem" -> derive[LSystem],
    "Camera" -> derive[Camera]
  )

  /** The unit of `typeName.field`, if it is an angle. */
  def of(typeName: String, field: String): Option[String] =
    byType.get(typeName).flatMap(_.get(field)).orElse(Option.when(field == "rotation")(Radians))
