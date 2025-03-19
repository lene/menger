package menger.objects.higher_d

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.badlogic.gdx.graphics.g3d.{Material, Model}
import menger.objects.LWJGLLoadChecker
import com.badlogic.gdx.math.Vector4

class RectMeshSuite extends AnyFlatSpec with Matchers:

  val face: QuadInfo = QuadInfo(VertexInfo(0, 0, 0), VertexInfo(1, 0, 0), VertexInfo(1, 1, 0), VertexInfo(0, 1, 0))
  val faces: Seq[QuadInfo] = Seq(face)
  val primitiveType = 0
  val material = Material()

  class RectMeshInstance extends RectMesh:
    lazy val modelData: Option[Model] =
      try Some(model(faces, primitiveType, material)) catch case _: Throwable => None

  "A RectMesh-derived object" should "be able to call model()" in:
    assume(LWJGLLoadChecker.loadingLWJGLSucceeds)
//    val instance = RectMeshInstance()
//    instance.model(faces, primitiveType, material) should not be null

//  it should "create a model of correct size" in new RectMeshInstance:
//    assume(loadingLWJGLSucceeds)
//    modelData.get.meshParts.size should be (1)
//
//  it should "create a model with the desired ID" in new RectMeshInstance:
//    assume(loadingLWJGLSucceeds)
//    modelData.get.meshParts.get(0).id should be ("sponge")
//
//  it should "create a model with the desired primitive type" in new RectMeshInstance:
//    assume(loadingLWJGLSucceeds)
//    modelData.get.meshParts.get(0).primitiveType should be (primitiveType)
//
//  it should "create a model with the right number of vertices" in new RectMeshInstance:
//    assume(loadingLWJGLSucceeds)
//    modelData.get.meshParts.get(0).mesh.getNumVertices should be (4)
//
//  it should "create a model with the right number of indices" in new RectMeshInstance:
//    assume(loadingLWJGLSucceeds)
//    modelData.get.meshParts.get(0).mesh.getNumIndices should be (4)

  "an edge's string representation" should "be correct" in:
    val edge = (Vector4(0, 0, 0, 0), Vector4(1, 1, 1, 1))
    edgeToString(edge) should include ("<0, 0, 0, 0>")
    edgeToString(edge) should include ("<1, 1, 1, 1>")

