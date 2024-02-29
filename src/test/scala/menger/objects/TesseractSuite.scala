package menger.objects

import com.badlogic.gdx.math.{Vector3, Vector4}
import org.scalatest.funsuite.AnyFunSuite

class TesseractSuite extends AnyFunSuite:

  test("tesseract has 16 vertices") {
    assert(Tesseract().vertices.size == 16)
  }

  test("all default tesseract vertices are +/-0.5") {
    assert(Tesseract().vertices.forall(v =>
      v.x == 0.5 || v.x == -0.5 && v.y == 0.5 || v.y == -0.5 &&
        v.z == 0.5 || v.z == -0.5 && v.w == 0.5 || v.w == -0.5
    ))
  }

  test("tesseract vertices scale with tesseract size") {
    Seq(2.0f, 10.0f, 1e8f, 0.5f, 1e-8f).foreach { size =>
      assert(Tesseract(2 * size).vertices.forall(v =>
        v.x == size || v.x == -size && v.y == size || v.y == -size &&
          v.z == size || v.z == -size && v.w == size || v.w == -size
      ))
    }
  }

  test("tesseract has 24 faces") {
    assert(Tesseract().faceIndices.size == 24)
  }

  test("first face of tesseract is correct") {
    assert(
      Tesseract(2).faces.head == (
        Vector4(-1.0,-1.0,-1.0,-1.0),
        Vector4(-1.0,-1.0,-1.0, 1.0),
        Vector4(-1.0,-1.0, 1.0, 1.0),
        Vector4(-1.0,-1.0, 1.0,-1.0)
      )
    )
  }

  test("last face of tesseract is correct") {
    assert(
      Tesseract(2).faces.last == (
        Vector4( 1.0, 1.0,-1.0,-1.0),
        Vector4( 1.0, 1.0,-1.0, 1.0),
        Vector4( 1.0, 1.0, 1.0, 1.0),
        Vector4( 1.0, 1.0, 1.0,-1.0)
      )
    )
  }

  test("tesseract has 32 edges") {
    assert(Tesseract().edges.size == 32)
  }

  test("every edge of a tesseract of unit size has length 2") {
    val edgeLengths = Tesseract().edges.map { case (a, b) => a.dst(b) }
    assert(edgeLengths.forall { _ == 1.0f })
  }

  test("TesseractProjection projectedFaceVertices size equals tesseract faces size") {
    val tp = TesseractProjection(Tesseract(), Projection(4, 1))
    assert(tp.projectedFaceVertices.size == tp.tesseract.faces.size)
  }

  test("untransformed projected tesseract has only 4 distinct coordinate values") {
    val tp = TesseractProjection(Tesseract(), Projection(4, 1))
    val coordinateValues = tp.projectedFaceVertices.flatMap(_.toList).flatMap(toList)
    assert(coordinateValues.distinct.size == 4)
  }

  test("untransformed projected tesseract has only 2 distinct absolute coordinate values") {
    val tp = TesseractProjection(Tesseract(), Projection(4, 1))
    val coordinateValues = tp.projectedFaceVertices.flatMap(_.toList).flatMap(toList).map(_.abs)
    assert(coordinateValues.distinct.size == 2)
  }

def toList(v: Vector3) = List(v.x, v.y, v.z)
