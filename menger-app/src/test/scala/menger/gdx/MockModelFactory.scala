package menger.gdx

import com.badlogic.gdx.graphics.g3d.Material
import com.badlogic.gdx.graphics.g3d.Model
import com.badlogic.gdx.graphics.g3d.model.Node
import com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder
import java.util.concurrent.atomic.AtomicReference

/**
 * Mock ModelFactory for testing without LibGDX/LWJGL initialization.
 *
 * Returns minimal stub models that satisfy type requirements.
 * Not suitable for actual rendering.
 */
class MockModelFactory extends ModelFactory:

  private val buildingModel: AtomicReference[Option[Int]] =
    new AtomicReference(None)

  private def createStubModel(material: Material): Model =
    // Create an empty Model without any meshes (avoids LibGDX initialization requirement)
    // This is sufficient for testing caching logic which doesn't inspect model contents
    val model = new Model()
    val node = new Node()
    model.nodes.add(node)
    model.materials.add(material)
    model

  def createSphere(
    width: Float,
    height: Float,
    depth: Float,
    divisionsU: Int,
    divisionsV: Int,
    material: Material,
    attributes: Long
  ): Model = createStubModel(material)

  def createBox(
    width: Float,
    height: Float,
    depth: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model = createStubModel(material)

  def createRect(
    x00: Float, y00: Float, z00: Float,
    x10: Float, y10: Float, z10: Float,
    x11: Float, y11: Float, z11: Float,
    x01: Float, y01: Float, z01: Float,
    normalX: Float, normalY: Float, normalZ: Float,
    primitiveType: Int,
    material: Material,
    attributes: Long
  ): Model = createStubModel(material)

  def begin(): Unit =
    buildingModel.set(Some(0))

  def part(id: String, primitiveType: Int, attributes: Long, material: Material): MeshPartBuilder =
    // Return a stub MeshPartBuilder that only implements rect() (the only method actually used)
    // The caching tests don't care about the actual mesh building, just the caching logic
    new Object with MeshPartBuilder:
      import com.badlogic.gdx.math.Vector3

      // Only implement the rect() overloads that are actually used in production code
      def rect(corner00: Vector3, corner10: Vector3, corner11: Vector3, corner01: Vector3, normal: Vector3): Unit = ()
      def rect(corner00: Vector3, corner10: Vector3, corner11: Vector3, corner01: Vector3): Unit = ()
      def rect(x00: Float, y00: Float, z00: Float, x10: Float, y10: Float, z10: Float, x11: Float, y11: Float, z11: Float, x01: Float, y01: Float, z01: Float, normalX: Float, normalY: Float, normalZ: Float): Unit = ()

      // All other methods are unused in tests - throw NotImplementedError if accidentally called
      def getAttributes(): com.badlogic.gdx.graphics.VertexAttributes = ???
      def getMeshPart(): com.badlogic.gdx.graphics.g3d.model.MeshPart = ???
      def ensureVertices(numVertices: Int): Unit = ???
      def getPrimitiveType(): Int = ???
      def setColor(color: com.badlogic.gdx.graphics.Color): Unit = ???
      def setColor(r: Float, g: Float, b: Float, a: Float): Unit = ???
      def setUVRange(u1: Float, v1: Float, u2: Float, v2: Float): Unit = ???
      def setUVRange(region: com.badlogic.gdx.graphics.g2d.TextureRegion): Unit = ???
      def setVertexTransform(transform: com.badlogic.gdx.math.Matrix4): Unit = ???
      def isVertexTransformationEnabled(): Boolean = ???
      def setVertexTransformationEnabled(enabled: Boolean): Unit = ???
      def lastIndex(): Int = ???
      def vertex(pos: Vector3, nor: Vector3, col: com.badlogic.gdx.graphics.Color, uv: com.badlogic.gdx.math.Vector2): Short = ???
      def vertex(info: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo): Short = ???
      def vertex(values: Float*): Short = ???
      def index(value: Short): Unit = ???
      def index(value1: Short, value2: Short): Unit = ???
      def index(value1: Short, value2: Short, value3: Short): Unit = ???
      def index(value1: Short, value2: Short, value3: Short, value4: Short): Unit = ???
      def index(value1: Short, value2: Short, value3: Short, value4: Short, value5: Short, value6: Short): Unit = ???
      def index(value1: Short, value2: Short, value3: Short, value4: Short, value5: Short, value6: Short, value7: Short, value8: Short): Unit = ???
      def line(p1: Vector3, p2: Vector3): Unit = ???
      def line(info1: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, info2: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo): Unit = ???
      def line(p1: Vector3, c1: com.badlogic.gdx.graphics.Color, p2: Vector3, c2: com.badlogic.gdx.graphics.Color): Unit = ???
      def line(x1: Float, y1: Float, z1: Float, x2: Float, y2: Float, z2: Float): Unit = ???
      def triangle(p1: Vector3, p2: Vector3, p3: Vector3): Unit = ???
      def triangle(info1: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, info2: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, info3: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo): Unit = ???
      def triangle(p1: Vector3, c1: com.badlogic.gdx.graphics.Color, p2: Vector3, c2: com.badlogic.gdx.graphics.Color, p3: Vector3, c3: com.badlogic.gdx.graphics.Color): Unit = ???
      def triangle(value1: Short, value2: Short, value3: Short): Unit = ???
      def addMesh(mesh: com.badlogic.gdx.graphics.Mesh): Unit = ???
      def addMesh(mesh: com.badlogic.gdx.graphics.Mesh, indexOffset: Int, numIndices: Int): Unit = ???
      def addMesh(vertices: Array[Float], indices: Array[Short]): Unit = ???
      def addMesh(vertices: Array[Float], indices: Array[Short], indexOffset: Int, numIndices: Int): Unit = ???
      def box(corner000: Vector3, corner010: Vector3, corner100: Vector3, corner110: Vector3, corner001: Vector3, corner011: Vector3, corner101: Vector3, corner111: Vector3): Unit = ???
      def box(transform: com.badlogic.gdx.math.Matrix4): Unit = ???
      def box(width: Float, height: Float, depth: Float): Unit = ???
      def circle(radius: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float): Unit = ???
      def circle(radius: Float, divisions: Int, center: Vector3, normal: Vector3): Unit = ???
      def circle(radius: Float, divisions: Int, center: Vector3, normal: Vector3, tangent: Vector3, binormal: Vector3): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, center: Vector3, normal: Vector3): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, center: Vector3, normal: Vector3, tangent: Vector3, binormal: Vector3): Unit = ???
      def ellipse(width: Float, height: Float, innerWidth: Float, innerHeight: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float): Unit = ???
      def ellipse(width: Float, height: Float, innerWidth: Float, innerHeight: Float, divisions: Int, center: Vector3, normal: Vector3): Unit = ???
      def ellipse(width: Float, height: Float, innerWidth: Float, innerHeight: Float, divisions: Int, center: Vector3, normal: Vector3, tangent: Vector3, binormal: Vector3): Unit = ???
      def cylinder(width: Float, height: Float, depth: Float, divisions: Int): Unit = ???
      def cylinder(width: Float, height: Float, depth: Float, divisions: Int, angleFrom: Float, angleTo: Float): Unit = ???
      def cylinder(width: Float, height: Float, depth: Float, divisions: Int, angleFrom: Float, angleTo: Float, close: Boolean): Unit = ???
      def cone(width: Float, height: Float, depth: Float, divisions: Int): Unit = ???
      def cone(width: Float, height: Float, depth: Float, divisions: Int, angleFrom: Float, angleTo: Float): Unit = ???
      def sphere(transform: com.badlogic.gdx.math.Matrix4, xDivisions: Float, yDivisions: Float): Unit = ???
      def sphere(width: Float, height: Float, depth: Float, xDivisions: Int, yDivisions: Int): Unit = ???
      def sphere(width: Float, height: Float, depth: Float, xDivisions: Int, yDivisions: Int, angleUFrom: Float, angleUTo: Float, angleVFrom: Float, angleVTo: Float): Unit = ???
      def sphere(transform: com.badlogic.gdx.math.Matrix4, xDivisions: Float, yDivisions: Float, zDivisions: Float, divisions: Int, divisionsV: Int): Unit = ???
      def sphere(transform: com.badlogic.gdx.math.Matrix4, xDivisions: Float, yDivisions: Float, zDivisions: Float, divisionsU: Int, divisionsV: Int, angleUFrom: Float, angleUTo: Float, angleVFrom: Float, angleVTo: Float): Unit = ???
      def capsule(radius: Float, height: Float, divisions: Int): Unit = ???
      def arrow(x1: Float, y1: Float, z1: Float, x2: Float, y2: Float, z2: Float, capLength: Float, stemThickness: Float, divisions: Int): Unit = ???
      def patch(corner00: Vector3, corner10: Vector3, corner11: Vector3, corner01: Vector3, divisionsU: Int, divisionsV: Int): Unit = ???
      def patch(v00: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v10: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v11: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v01: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, divisionsU: Int, divisionsV: Int): Unit = ???
      def patch(corner00: Vector3, corner10: Vector3, corner11: Vector3, corner01: Vector3, normal: Vector3, divisionsU: Int, divisionsV: Int): Unit = ???
      def patch(x00: Float, y00: Float, z00: Float, x10: Float, y10: Float, z10: Float, x11: Float, y11: Float, z11: Float, x01: Float, y01: Float, z01: Float, normalX: Float, normalY: Float, normalZ: Float, divisionsU: Int, divisionsV: Int): Unit = ???
      def line(v1: Short, v2: Short): Unit = ???
      def getVertexTransform(out: com.badlogic.gdx.math.Matrix4): com.badlogic.gdx.math.Matrix4 = ???
      def rect(v1: Short, v2: Short, v3: Short, v4: Short): Unit = ???
      def rect(v00: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v10: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v11: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v01: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo): Unit = ???
      def rect(x00: Float, y00: Float, z00: Float, x10: Float, y10: Float, z10: Float, x11: Float, y11: Float, z11: Float, x01: Float, y01: Float, z01: Float, normalX: Float, normalY: Float, normalZ: Float, colorX: Int, colorY: Int): Unit = ???
      def addMesh(meshPart: com.badlogic.gdx.graphics.g3d.model.MeshPart): Unit = ???
      def box(v000: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v010: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v100: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v110: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v001: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v011: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v101: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo, v111: com.badlogic.gdx.graphics.g3d.utils.MeshPartBuilder.VertexInfo): Unit = ???
      def box(x: Float, y: Float, z: Float, w: Float, h: Float, d: Float): Unit = ???
      def circle(radius: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, tangentX: Float, tangentY: Float, tangentZ: Float, binormalX: Float, binormalY: Float, binormalZ: Float): Unit = ???
      def circle(radius: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def circle(radius: Float, divisions: Int, center: Vector3, normal: Vector3, angleFrom: Float, angleTo: Float): Unit = ???
      def circle(radius: Float, divisions: Int, center: Vector3, normal: Vector3, tangent: Vector3, binormal: Vector3, angleFrom: Float, angleTo: Float): Unit = ???
      def circle(radius: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, tangentX: Float, tangentY: Float, tangentZ: Float, binormalX: Float, binormalY: Float, binormalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, tangentX: Float, tangentY: Float, tangentZ: Float, binormalX: Float, binormalY: Float, binormalZ: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, center: Vector3, normal: Vector3, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, center: Vector3, normal: Vector3, tangent: Vector3, binormal: Vector3, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, tangentX: Float, tangentY: Float, tangentZ: Float, binormalX: Float, binormalY: Float, binormalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, innerWidth: Float, innerHeight: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, tangentX: Float, tangentY: Float, tangentZ: Float, binormalX: Float, binormalY: Float, binormalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def ellipse(width: Float, height: Float, innerWidth: Float, innerHeight: Float, divisions: Int, centerX: Float, centerY: Float, centerZ: Float, normalX: Float, normalY: Float, normalZ: Float, angleFrom: Float, angleTo: Float): Unit = ???
      def ensureCapacity(numVertices: Int, numIndices: Int): Unit = ???
      def ensureIndices(numIndices: Int): Unit = ???
      def ensureRectangleIndices(numRectangles: Int): Unit = ???
      def ensureTriangleIndices(numTriangles: Int): Unit = ???

  def end(): Model =
    buildingModel.get().getOrElse(
      sys.error("end() called without begin()")
    )
    buildingModel.set(None)

    // Create a model without meshes (for testing purposes)
    // The caching logic doesn't depend on model contents
    val model = new Model()
    val node = new Node()
    model.nodes.add(node)
    model
