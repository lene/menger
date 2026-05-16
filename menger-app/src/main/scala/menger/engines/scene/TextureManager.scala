package menger.engines.scene

import java.nio.file.Paths

import scala.util.Failure
import scala.util.Success

import com.typesafe.scalalogging.LazyLogging
import menger.ObjectSpec
import menger.TextureLoader
import menger.optix.OptiXRenderer

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
    // Collect unique texture filenames from all specs (albedo + PBR maps)
    val textureFilenames = (
      specs.flatMap(_.texture) ++
      specs.flatMap(_.normalMap) ++
      specs.flatMap(_.roughnessMap)
    ).distinct

    if textureFilenames.isEmpty then
      Map.empty
    else
      logger.info(s"Loading ${textureFilenames.length} texture(s)")

      textureFilenames.flatMap { filename =>
        if filename.toLowerCase.endsWith(".hdr") then
          val resolvedPath =
            if Paths.get(filename).isAbsolute then filename
            else Paths.get(textureDir).resolve(filename).toString
          val idx = renderer.uploadTextureFromFile(resolvedPath)
          if idx >= 0 then
            logger.debug(s"Uploaded HDR texture '$filename' as index $idx")
            Some(filename -> idx)
          else
            logger.error(s"Failed to upload HDR texture '$filename'")
            None
        else
          TextureLoader.load(filename, textureDir) match
            case Success(textureData) =>
              renderer.uploadTexture(
                textureData.name,
                textureData.data,
                textureData.width,
                textureData.height
              ) match
                case Success(index) =>
                  logger.debug(s"Uploaded texture '$filename' as index $index")
                  Some(filename -> index)
                case Failure(e) =>
                  logger.error(s"Failed to upload texture '$filename': ${e.getMessage}")
                  None
            case Failure(e) =>
              logger.error(s"Failed to load texture '$filename': ${e.getMessage}")
              None
      }.toMap
