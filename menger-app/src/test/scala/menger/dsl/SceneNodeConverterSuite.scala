package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import menger.engines.SceneConverter

class SceneNodeConverterSuite extends AnyFlatSpec with Matchers:

  private val fallbackCaustics = Caustics.Disabled.toCausticsConfig

  // --- flattenNode ---

  "SceneConverter.flattenNode" should "produce one ObjectSpec from a single leaf node" in:
    val node = SceneNode.leaf(Sphere(Vec3(1f, 2f, 3f), Material.Glass))
    val specs = SceneConverter.flattenNode(node, Transform.Identity, None)
    specs should have length 1
    specs.head.objectType shouldBe "sphere"

  it should "preserve geometry position when transform is identity" in:
    val node = SceneNode.leaf(Sphere(Vec3(3f, 0f, 0f)))
    val specs = SceneConverter.flattenNode(node, Transform.Identity, None)
    specs.head.x shouldBe 3f +- 1e-5f
    specs.head.y shouldBe 0f +- 1e-5f
    specs.head.z shouldBe 0f +- 1e-5f

  it should "apply node translation to geometry position" in:
    val node = SceneNode(transform = Transform.at(Vec3(5f, 0f, 0f)), geometry = Some(Sphere()))
    val specs = SceneConverter.flattenNode(node, Transform.Identity, None)
    specs.head.x shouldBe 5f +- 1e-5f

  it should "apply node scale to geometry size" in:
    val node = SceneNode(transform = Transform.scaled(2f), geometry = Some(Sphere(size = 1f)))
    val specs = SceneConverter.flattenNode(node, Transform.Identity, None)
    specs.head.size shouldBe 2f +- 1e-5f

  it should "accumulate parent world transform with node transform" in:
    val parentWorld = Transform(translation = Vec3(1f, 0f, 0f), scale = 2f)
    val node = SceneNode(transform = Transform.at(Vec3(1f, 0f, 0f)), geometry = Some(Sphere()))
    val specs = SceneConverter.flattenNode(node, parentWorld, None)
    // world position = parentWorld.translation + parentWorld.scale * node.transform.translation + world.scale * geometry.pos
    // = (1,0,0) + 2*(1,0,0) + 2*1*(0,0,0) = (3,0,0)
    specs.head.x shouldBe 3f +- 1e-4f

  it should "collect specs from all leaf nodes in a group" in:
    val root = SceneNode.group(
      SceneNode.leaf(Sphere()),
      SceneNode.leaf(Cube())
    )
    val specs = SceneConverter.flattenNode(root, Transform.Identity, None)
    specs should have length 2

  it should "inherit material from parent node when geometry has no material" in:
    val chrome = Material.Chrome
    val group = SceneNode(
      material = Some(chrome),
      children = List(SceneNode.leaf(Sphere()))
    )
    val specs = SceneConverter.flattenNode(group, Transform.Identity, None)
    specs should have length 1
    specs.head.material shouldBe defined
    specs.head.material.get.metallic shouldBe chrome.metallic +- 1e-5f

  it should "child node material overrides inherited material" in:
    val chrome = Material.Chrome
    val glass  = Material.Glass
    val group = SceneNode(
      material = Some(chrome),
      children = List(
        SceneNode(material = Some(glass), geometry = Some(Sphere()))
      )
    )
    val specs = SceneConverter.flattenNode(group, Transform.Identity, None)
    specs.head.material shouldBe defined
    specs.head.material.get.ior shouldBe glass.ior +- 1e-5f

  it should "geometry with own material is not overridden by node material" in:
    val chrome = Material.Chrome
    val glass  = Material.Glass
    val group = SceneNode(
      material = Some(chrome),
      children = List(SceneNode.leaf(Sphere(material = Some(glass))))
    )
    val specs = SceneConverter.flattenNode(group, Transform.Identity, None)
    specs.head.material shouldBe defined
    specs.head.material.get.ior shouldBe glass.ior +- 1e-5f

  it should "handle deep tree with transform and material inheritance" in:
    val parentGroup = SceneNode(
      transform = Transform.at(Vec3(10f, 0f, 0f)),
      material = Some(Material.Gold),
      children = List(
        SceneNode(
          transform = Transform.at(Vec3(1f, 0f, 0f)),
          children = List(SceneNode.leaf(Sphere()))
        )
      )
    )
    val specs = SceneConverter.flattenNode(parentGroup, Transform.Identity, None)
    specs should have length 1
    specs.head.x shouldBe 11f +- 1e-4f
    specs.head.material shouldBe defined

  // --- Scene with root SceneNode ---

  "SceneConverter.convert" should "convert a scene with a root SceneNode" in:
    val root  = SceneNode.leaf(Sphere(Vec3(1f, 2f, 3f), Material.Glass))
    val scene = Scene(Camera.Default, root = Some(root))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    result.scene.objectSpecs shouldBe defined
    val specs = result.scene.objectSpecs.get
    specs should have length 1
    specs.head.objectType shouldBe "sphere"
    specs.head.x shouldBe 1f +- 1e-5f

  it should "prefer root SceneNode over flat objects list when both are present" in:
    val root   = SceneNode.leaf(Sphere(Vec3(0f, 0f, 0f), Material.Glass))
    val flat   = Sphere(Vec3(99f, 0f, 0f), Material.Chrome)
    val scene  = Scene(Camera.Default, objects = List(flat), root = Some(root))
    val result = SceneConverter.convert(scene, fallbackCaustics)
    val specs  = result.scene.objectSpecs.get
    specs should have length 1
    specs.head.x shouldBe 0f +- 1e-5f

  "Scene with root" should "be constructible with only a root node (no flat objects)" in:
    val root  = SceneNode.leaf(Sphere(Material.Glass))
    val scene = Scene(Camera.Default, root = Some(root))
    scene.root shouldBe defined
    scene.objects shouldBe empty
