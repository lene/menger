package menger.dsl

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SceneNodeSuite extends AnyFlatSpec with Matchers:

  private val glass   = Material.Glass
  private val chrome  = Material.Chrome
  private val sphere  = Sphere(Vec3(1f, 2f, 3f), glass)
  private val cube    = Cube(Vec3(-1f, 0f, 0f), chrome)

  "SceneNode" should "default to identity transform, no material, no geometry, no children" in:
    val node = SceneNode()
    node.transform shouldBe Transform.Identity
    node.material shouldBe None
    node.geometry shouldBe None
    node.children shouldBe Nil

  "SceneNode.leaf" should "create a leaf node with geometry" in:
    val node = SceneNode.leaf(sphere)
    node.geometry shouldBe Some(sphere)
    node.children shouldBe Nil
    node.transform shouldBe Transform.Identity

  it should "accept a transform" in:
    val t = Transform.at(Vec3(5f, 0f, 0f))
    val node = SceneNode.leaf(t, sphere)
    node.transform shouldBe t
    node.geometry shouldBe Some(sphere)

  "SceneNode.group" should "create a group with children and no geometry" in:
    val leaf1 = SceneNode.leaf(sphere)
    val leaf2 = SceneNode.leaf(cube)
    val group = SceneNode.group(leaf1, leaf2)
    group.geometry shouldBe None
    group.children should have length 2
    group.children should contain(leaf1)
    group.children should contain(leaf2)

  it should "accept a transform for the group" in:
    val t = Transform.at(Vec3(10f, 0f, 0f))
    val leaf = SceneNode.leaf(sphere)
    val group = SceneNode.group(t, leaf)
    group.transform shouldBe t
    group.children should have length 1

  it should "accept a material for the group" in:
    val leaf = SceneNode.leaf(Sphere())
    val group = SceneNode.group(chrome, leaf)
    group.material shouldBe Some(chrome)

  "SceneNode.allLeafGeometry" should "return geometry from a single leaf node" in:
    val node = SceneNode.leaf(sphere)
    node.allLeafGeometry should have length 1
    node.allLeafGeometry.head shouldBe sphere

  it should "return empty for a node with no geometry and no children" in:
    SceneNode().allLeafGeometry shouldBe empty

  it should "collect geometry from all leaves in a tree" in:
    val leaf1 = SceneNode.leaf(sphere)
    val leaf2 = SceneNode.leaf(cube)
    val root  = SceneNode.group(leaf1, leaf2)
    root.allLeafGeometry should have length 2

  it should "collect from deeply nested trees" in:
    val deep = SceneNode.group(
      SceneNode.group(
        SceneNode.leaf(sphere),
        SceneNode.leaf(cube)
      ),
      SceneNode.leaf(sphere)
    )
    deep.allLeafGeometry should have length 3
