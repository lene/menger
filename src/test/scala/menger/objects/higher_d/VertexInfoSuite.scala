package menger.objects.higher_d

import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder
import com.badlogic.gdx.math.Vector3
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VertexInfoSuite extends AnyFlatSpec with Matchers:

  val vertex: MeshPartBuilder.VertexInfo = VertexInfo(Vector3(1, 2, 3))

  "VertexInfo from Vector3" should "set a position" in:
    vertex.hasPosition should be (true)

  it should "keep set position" in:
    vertex.position should be (Vector3(1, 2, 3))

  it should "not set normal" in:
    vertex.hasNormal should be (false)

  it should "not set color" in:
    vertex.hasColor should be (false)

