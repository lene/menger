package menger.objects.higher_d

import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec

extension(v: Vector3)
  def toList: List[Float] = List(v.x, v.y, v.z)

class RotatedProjectionSuite extends AnyFlatSpec:

  trait ProjectedTesseract:
    val tesseract: RotatedProjection = RotatedProjection(Tesseract(), Projection(4, 1))
    val coordinateValues: Seq[Float] = tesseract.projectedFaceVertices.flatMap(_.toList).flatMap(_.toList)

  "A RotatedProjection" should "be able to be created from a Projection alone" in:
    RotatedProjection(Tesseract(), Projection(4, 1))

  it should "be able to be created from a Projection and Rotation" in:
    RotatedProjection(Tesseract(), Projection(4, 1), Rotation(90, 0, 0))

  "A Tesseract's Projection's projectedFaceVertices" should "have equal size to faces" in new ProjectedTesseract:
    assert(tesseract.projectedFaceVertices.size == tesseract.object4D.faces.size)

  it should "have only 4 distinct coordinate values if not rotated" in new ProjectedTesseract:
    assert(coordinateValues.distinct.size == 4)

  it should "have only 2 distinct absolute coordinate values if not rotated" in new ProjectedTesseract:
    assert(coordinateValues.map(_.abs).distinct.size == 2)

  "A RotatedProjection's projectedFaceInfo" should "have equal size to faces" in new ProjectedTesseract:
    assert(tesseract.projectedFaceInfo.size == tesseract.object4D.faces.size)

  it should "have a position set for all vertices" in new ProjectedTesseract:
    assert(tesseract.projectedFaceInfo.forall(v => v.toList.forall(_.hasPosition)))

  it should "have the correct position set for all vertices" in new ProjectedTesseract:
    assert(
      tesseract.projectedFaceVertices.zip(tesseract.projectedFaceInfo).forall {
        case (vertices, infos) => vertices.toList.zip(infos.toList).forall {
          case (vertex, info) => vertex == info.position
        }
      }
    )
