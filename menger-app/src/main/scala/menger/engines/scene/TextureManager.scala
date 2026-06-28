package menger.engines.scene

import java.nio.file.Path
import java.nio.file.Paths

import scala.util.Failure
import scala.util.Success
import scala.util.Try
import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging
import io.github.lene.optix.OptiXRenderer
import io.github.lene.optix.TextureUploadException
import menger.ObjectSpec
import menger.TextureData
import menger.TextureLoader
import menger.geometry.VideoLoader
import menger.video.EnvMapVideo
import menger.video.VideoTexture

/**
 * Manager for loading and uploading textures to OptiX renderer.
 *
 * Handles texture loading workflow:
 * 1. Collect unique texture filenames from object specs
 * 2. Load texture data from files
 * 3. Upload to OptiX renderer
 * 4. Return filename → texture index mapping
 *
 * Texture loading failures are logged but don't stop scene creation -
 * objects without textures will use materials only.
 */
object TextureManager extends LazyLogging:
  type VideoTextureSlotObserver = (VideoTexture, Int) => Unit
  type VideoTextureSlotProvider = VideoTexture => Option[Int]

  private val videoTextureSlotObserver = new ThreadLocal[Option[VideoTextureSlotObserver]]:
    override def initialValue(): Option[VideoTextureSlotObserver] = None

  private val videoTextureSlotProvider = new ThreadLocal[Option[VideoTextureSlotProvider]]:
    override def initialValue(): Option[VideoTextureSlotProvider] = None

  def withVideoTextureSlotObserver[A](observer: VideoTextureSlotObserver)(body: => A): A =
    val previousObserver = videoTextureSlotObserver.get()
    videoTextureSlotObserver.set(Some(observer))
    try body
    finally videoTextureSlotObserver.set(previousObserver)

  def withVideoTextureSlotProvider[A](provider: VideoTextureSlotProvider)(body: => A): A =
    val previousProvider = videoTextureSlotProvider.get()
    videoTextureSlotProvider.set(Some(provider))
    try body
    finally videoTextureSlotProvider.set(previousProvider)

  /**
   * Load all textures referenced by object specs and upload to renderer.
   *
   * @param specs List of object specifications (may contain texture references)
   * @param renderer OptiX renderer to upload textures to
   * @param textureDir Directory containing texture files
   * @return Map from texture filename to texture index (for successfully loaded textures)
   */
  def loadTextures(
    specs: List[ObjectSpec],
    renderer: OptiXRenderer,
    textureDir: String
  ): Map[String, Int] =
    val staticTextureFilenames = (
      specs.flatMap(_.texture) ++ specs.flatMap(_.normalMap) ++ specs.flatMap(_.roughnessMap)
    ).distinct
    val videoTextures = specs.flatMap(_.videoTexture).distinctBy(_.textureKey)
    val textureCount = staticTextureFilenames.length + videoTextures.length

    if textureCount == 0 then
      Map.empty
    else
      logger.info(s"Loading $textureCount texture(s)")

      val staticTextureIndices = staticTextureFilenames.flatMap { filename =>
        loadStaticTexture(filename, renderer, textureDir)
      }
      val videoTextureIndices = videoTextures.flatMap { videoTexture =>
        loadInitialVideoTexture(videoTexture, renderer, textureDir)
      }
      (staticTextureIndices ++ videoTextureIndices).toMap

  private def loadStaticTexture(
    filename: String,
    renderer: OptiXRenderer,
    textureDir: String
  ): Option[(String, Int)] =
    if filename.toLowerCase.endsWith(".hdr") then
      val resolvedPath = resolveTexturePath(filename, textureDir).toString
      try
        val idx = renderer.uploadTextureFromFile(resolvedPath)
        logger.debug(s"Uploaded HDR texture '$filename' as index $idx")
        Some(filename -> idx)
      catch
        case e: TextureUploadException =>
          logger.error(s"Failed to upload HDR texture '$filename': ${e.getMessage}")
          None
    else
      TextureLoader.load(filename, textureDir) match
        case Success(textureData) =>
          uploadTextureData(textureData, renderer)
        case Failure(e) =>
          logger.error(s"Failed to load texture '$filename': ${e.getMessage}")
          None

  private def loadInitialVideoTexture(
    videoTexture: VideoTexture,
    renderer: OptiXRenderer,
    textureDir: String
  ): Option[(String, Int)] =
    videoTextureSlotProvider.get().flatMap(_(videoTexture)) match
      case Some(textureIndex) =>
        recordVideoTextureSlot(videoTexture, textureIndex)
        Some(videoTexture.textureKey -> textureIndex)
      case None =>
        loadInitialVideoTextureData(videoTexture, textureDir) match
          case Success(textureData) =>
            val uploadedTexture = uploadTextureData(textureData, renderer)
            uploadedTexture.foreach { case (_, textureIndex) =>
              recordVideoTextureSlot(videoTexture, textureIndex)
            }
            uploadedTexture
          case Failure(e) =>
            logger.error(s"Failed to load video texture '${videoTexture.path}': ${e.getMessage}")
            None

  def loadInitialEnvMapVideo(
    envMapVideo: EnvMapVideo,
    renderer: OptiXRenderer,
    textureDir: String
  ): Option[Int] =
    loadInitialEnvMapVideoData(envMapVideo, textureDir) match
      case Success(textureData) =>
        uploadTextureData(textureData, renderer).map(_._2)
      case Failure(e) =>
        logger.error(s"Failed to load environment-map video '${envMapVideo.path}': ${e.getMessage}")
        None

  private[scene] def loadInitialVideoTextureData(
    videoTexture: VideoTexture,
    textureDir: String
  ): Try[TextureData] =
    val resolvedPath = resolveTexturePath(videoTexture.path, textureDir)
    loadVideoTextureData:
      val loader = new VideoLoader(resolvedPath.toString)
      try
        TextureData(
          videoTexture.textureKey,
          loader.frameAt(videoTexture.playback.startOffset),
          loader.width,
          loader.height
        )
      finally loader.close()

  private[scene] def loadInitialEnvMapVideoData(
    envMapVideo: EnvMapVideo,
    textureDir: String
  ): Try[TextureData] =
    val resolvedPath = resolveTexturePath(envMapVideo.path, textureDir)
    loadVideoTextureData:
      val loader = new VideoLoader(resolvedPath.toString)
      try
        validateEquirectangularDimensions(
          loader.width,
          loader.height,
          envMapVideo.path
        )
        TextureData(
          envMapVideo.textureKey,
          loader.frameAt(envMapVideo.playback.startOffset),
          loader.width,
          loader.height
        )
      finally loader.close()

  private def uploadTextureData(
    textureData: TextureData,
    renderer: OptiXRenderer
  ): Option[(String, Int)] =
    try
      val index = renderer.uploadTexture(
        textureData.name,
        textureData.data,
        textureData.width,
        textureData.height
      )
      logger.debug(s"Uploaded texture '${textureData.name}' as index $index")
      Some(textureData.name -> index)
    catch
      case e: Exception =>
        logger.error(s"Failed to upload texture '${textureData.name}': ${e.getMessage}")
        None

  private def resolveTexturePath(filename: String, textureDir: String): Path =
    val filePath = Paths.get(filename)
    if filePath.isAbsolute then filePath
    else Paths.get(textureDir).resolve(filename)

  private def recordVideoTextureSlot(videoTexture: VideoTexture, textureIndex: Int): Unit =
    videoTextureSlotObserver.get().foreach(_(videoTexture, textureIndex))

  private def loadVideoTextureData(body: => TextureData): Try[TextureData] =
    try Success(body)
    catch
      case NonFatal(e)      => Failure(e)
      case e: LinkageError => Failure(e)

  private[engines] def validateEquirectangularDimensions(
    width: Int,
    height: Int,
    path: String
  ): Unit =
    require(
      width == height * 2,
      s"Environment-map video '$path' must be equirectangular 2:1, got ${width}x$height"
    )
