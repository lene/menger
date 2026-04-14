package menger.engines

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class VideoEncoderSuite extends AnyFlatSpec with Matchers:

  "VideoEncoder.buildArgs" should "produce libx264 args for .mp4 output" in:
    val args = VideoEncoder.buildArgs("frames/frame_%04d.png", "output.mp4", fps = 24, quality = 12)
    args should contain("ffmpeg")
    args should contain("-framerate")
    args should contain("24")
    args should contain("-i")
    args should contain("frames/frame_%04d.png")
    args should contain("-c:v")
    args should contain("libx264")
    args should contain("-qp")
    args should contain("12")
    args should contain("-pix_fmt")
    args should contain("yuv420p")
    args should contain("output.mp4")

  it should "produce hevc_nvenc args for .mkv output" in:
    val args = VideoEncoder.buildArgs("frames/frame_%04d.png", "output.mkv", fps = 30, quality = 12)
    args should contain("hevc_nvenc")
    args should contain("-qp")
    args should contain("12")
    args should contain("output.mkv")
    args should not contain "libx264"
    args should not contain "yuv420p"

  it should "include -y flag to overwrite without prompting" in:
    val args = VideoEncoder.buildArgs("frame_%04d.png", "out.mp4", fps = 24, quality = 12)
    args should contain("-y")

  it should "pass the fps value correctly" in:
    val args = VideoEncoder.buildArgs("frame_%04d.png", "out.mkv", fps = 60, quality = 18)
    val framerateIdx = args.indexOf("-framerate")
    framerateIdx should be >= 0
    args(framerateIdx + 1) shouldBe "60"

  it should "throw for unsupported extension" in:
    a[RuntimeException] should be thrownBy
      VideoEncoder.buildArgs("frame_%04d.png", "output.webm", fps = 24, quality = 12)

  it should "throw for no extension" in:
    a[RuntimeException] should be thrownBy
      VideoEncoder.buildArgs("frame_%04d.png", "output", fps = 24, quality = 12)

  "VideoEncoder.checkAvailable" should "succeed for .mp4 on this system (ffmpeg + libx264 present)" in:
    noException should be thrownBy VideoEncoder.checkAvailable("output.mp4")

  it should "succeed for .mkv on this system (ffmpeg + hevc_nvenc present)" in:
    noException should be thrownBy VideoEncoder.checkAvailable("output.mkv")

  it should "throw for unsupported extension" in:
    a[RuntimeException] should be thrownBy VideoEncoder.checkAvailable("output.avi")
