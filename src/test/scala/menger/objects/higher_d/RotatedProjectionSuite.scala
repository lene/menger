package menger.objects.higher_d

import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo
import com.badlogic.gdx.math.{Matrix4, Vector3}
import menger.{MengerCLIOptions, RotationProjectionParameters}
import menger.objects.{Builder, LWJGLLoadChecker}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

extension(v: Vector3)
  def toList: List[Float] = List(v.x, v.y, v.z)

class RotatedProjectionSuite extends AnyFlatSpec with Matchers with CustomMatchers:

  trait ProjectedTesseract:
    val tesseract: RotatedProjection = RotatedProjection(Tesseract(), Projection(4, 1))
    val coordinateValues: Seq[Float] = tesseract.projectedFaceVertices.flatMap(_.toList).flatMap(_.toList)

  "A RotatedProjection" should "be able to be created from a Projection alone" in:
    RotatedProjection(Tesseract(), Projection(4, 1))

  it should "be able to be created from a Projection and Rotation" in:
    RotatedProjection(Tesseract(), Projection(4, 1), Rotation(90, 0, 0))

  it should "be created without error with 3D and 4D rotations from CLI args" in :
    val rotProjParams = RotationProjectionParameters(MengerCLIOptions(Seq[String](
      "--rot-x", "90", "--rot-z-w", "90"
    )))
    RotatedProjection(Tesseract(), rotProjParams, Builder.WHITE_MATERIAL, GL20.GL_TRIANGLES)

  it should "be created correctly with 3D and 4D rotations from CLI args" in :
    val rotProjParams = RotationProjectionParameters(MengerCLIOptions(Seq[String](
      "--rot-x", "90", "--rot-z-w", "90"
    )))
    RotatedProjection(
      Tesseract(), rotProjParams, Builder.WHITE_MATERIAL, GL20.GL_TRIANGLES
    ).rotation.transformationMatrix should epsilonEqual(
    Matrix4(Array[Float](
      0, -1, 0, 0,
      1, 0, 0, 0,
      0, 0, 0, -1,
      0, 0, 1, 0
    )))

  "A Tesseract's Projection's projectedFaceVertices" should "have equal size to faces" in new ProjectedTesseract:
    tesseract.projectedFaceVertices should have size tesseract.object4D.faces.size

  it should "have only 4 distinct coordinate values if not rotated" in new ProjectedTesseract:
    coordinateValues.distinct should have size 4

  it should "have only 2 distinct absolute coordinate values if not rotated" in new ProjectedTesseract:
    coordinateValues.map(_.abs).distinct should have size 2

  "A RotatedProjection's projectedFaceInfo" should "have equal size to faces" in new ProjectedTesseract:
    tesseract.projectedFaceInfo should have size tesseract.object4D.faces.size

  it should "have a position set for all vertices" in new ProjectedTesseract:
    forAll(tesseract.projectedFaceInfo) {
      p => forAll(p.toList) { _.hasPosition should be(true) }
    }

  it should "have the correct position set for all vertices" in new ProjectedTesseract:
    forAll(tesseract.projectedFaceVertices.zip(tesseract.projectedFaceInfo)) {
      case (vertices, infos) => forAll(vertices.toList.zip(infos.toList)) {
        case (vertex, info) => vertex should be (info.position)
      }
    }
