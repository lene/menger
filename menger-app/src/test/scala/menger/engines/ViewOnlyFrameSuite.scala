package menger.engines

import scala.util.Using

import io.github.lene.optix.Material
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.testing.OptiXTestHooks
import menger.common.Color
import menger.common.ImageSize
import menger.common.Vector
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** menger#86's fitness function: a frame that only moves the camera (`FrameChange.Unchanged`)
  * leaves the instances alone and costs no full IAS build; the old per-frame
  * clearAllInstances + rebuild cost one full build every frame. */
class ViewOnlyFrameSuite extends AnyFlatSpec with Matchers:

  private val imageSize = ImageSize(64, 48)
  private val Frames = 5
  private val material = Material(Color(0.9f, 0.1f, 0.1f, 1f), 1.0f)
  private val sphere = Array(0.5f, 0f, 0f, 0f, 0f, 0.5f, 0f, 0f, 0f, 0f, 0.5f, 0f)

  private def lookFrom(r: OptiXRenderer, frame: Int): Unit =
    r.setCamera(Vector[3](frame * 0.1f, 0f, 4f), Vector[3](0f, 0f, 0f), Vector[3](0f, 1f, 0f), 60f)

  /** Full IAS builds over `Frames` frames after the first, each frame running `perFrame`. */
  private def fullBuilds(perFrame: OptiXRenderer => Unit): Long =
    Using.resource(new OptiXRenderer()): r =>
      r.initialize() shouldBe true
      r.setLight(Vector[3](-0.5f, -0.5f, -1f), 1.0f)
      val _ = r.addSphereInstance(sphere, material)
      lookFrom(r, 0)
      val _ = r.render(imageSize)
      val before = OptiXTestHooks.iasUpdateCounts(r).fullBuilds
      (1 to Frames).foreach { frame =>
        perFrame(r)
        lookFrom(r, frame)
        val _ = r.render(imageSize)
      }
      OptiXTestHooks.iasUpdateCounts(r).fullBuilds - before

  "A view-only frame" should "cost no full IAS build" in:
    assume(OptiXRenderer.isLibraryLoaded, "OptiX native library not loaded")
    fullBuilds(_ => ()) shouldBe 0L

  "A rebuilt frame" should "cost one full IAS build, which is what view-only frames now skip" in:
    assume(OptiXRenderer.isLibraryLoaded, "OptiX native library not loaded")
    fullBuilds { r =>
      r.clearAllInstances()
      val _ = r.addSphereInstance(sphere, material)
    } shouldBe Frames.toLong
