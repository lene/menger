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
//
//  "A RectMesh-derived object" should "be able to call model()" in new RectMeshInstance:
//    assume(LWJGLLoadChecker.loadingLWJGLSucceeds)
