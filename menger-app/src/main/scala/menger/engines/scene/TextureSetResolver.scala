package menger.engines.scene

import java.nio.file.{Files, Path}
import scala.util.{Failure, Success, Try, Using}
import scala.jdk.CollectionConverters._

/** Describes a resolved PBR texture set from a directory. */
case class ResolvedTextureSet(
  color: Option[Path],
  normal: Option[Path],
  roughness: Option[Path],
  metallic: Option[Path],
  ao: Option[Path],
  height: Option[Path],
  normalNeedsDXConversion: Boolean = false
)

/** Detects and resolves PBR texture sets from ambientCG and Poly Haven conventions.
  *
  * Given a directory, scans for image files and classifies them by map type using
  * known naming conventions. Uses a heuristic: the first convention that matches
  * at least 3 file types wins (prevents mixed-convention false matches).
  *
  * Pure Scala — unit-testable without GPU.
  */
object TextureSetResolver:

  private val ImageExtensions = Set(".png", ".jpg", ".jpeg", ".tga", ".hdr", ".bmp")

  /** Map type classification result. */
  private enum MapType:
    case Color, NormalGL, NormalDX, Roughness, Metallic, AO, Height, Unknown

  /** Convention-specific pattern matching. */
  private trait Convention:
    def classify(name: String): MapType

  /** Poly Haven: lowercase with underscores, e.g. `my_material_diff.png`. */
  private object PolyHaven extends Convention:
    def classify(name: String): MapType = name.toLowerCase match
      case n if n.contains("_diff")   || n.contains("_albedo")   => MapType.Color
      case n if n.contains("_nor_gl") => MapType.NormalGL
      case n if n.contains("_nor_dx") => MapType.NormalDX
      case n if n.contains("_rough")  => MapType.Roughness
      case n if n.contains("_metal")  => MapType.Metallic
      case n if n.contains("_ao")     => MapType.AO
      case n if n.contains("_disp")   => MapType.Height
      case _ => MapType.Unknown

  /** ambientCG: PascalCase, e.g. `PavingStones067_Color_4K.jpg`.
    * Resolution suffix is stripped before classification.
    */
  private object AmbientCG extends Convention:
    private val resSuffixes = Seq("_1K", "_2K", "_4K", "_8K", "_16K")
    def classify(name: String): MapType =
      val stripped = name.toLowerCase
      val base = resSuffixes.foldLeft(stripped)((n, s) => n.replace(s, ""))
      base match
        case n if n.contains("_color")  || n.contains("_basecolor") => MapType.Color
        case n if n.contains("_normalgl") => MapType.NormalGL
        case n if n.contains("_normaldx") => MapType.NormalDX
        case n if n.contains("_roughness") => MapType.Roughness
        case n if n.contains("_metalness") => MapType.Metallic
        case n if n.contains("_ambientocclusion") => MapType.AO
        case n if n.contains("_displacement") => MapType.Height
        case _ => MapType.Unknown

  /** Detect which convention a directory uses.
    * Returns the convention that classifies the most files into known types.
    */
  private def detectConvention(files: Seq[Path]): Option[Convention] =
    val names = files.map(f => f.getFileName.toString.toLowerCase)
    val conventions: Seq[Convention] = Seq(PolyHaven, AmbientCG)
    conventions
      .map(c => c -> names.count(n => c.classify(n) != MapType.Unknown))
      .filter(_._2 >= 3) // At least 3 known types to trust detection
      .sortBy(-_._2)
      .headOption
      .map(_._1)

  /** Extract resolution from a filename.
    * ambientCG: `_4K`, `_2K`, etc.
    * Returns None if not found.
    */
  private def extractResolution(name: String): Option[Int] =
    val resPattern = """_(\d+)K""".r.unanchored
    name match
      case resPattern(k) => Try(k.toInt * 1024).toOption // "2K" → 2048
      case _ => None

  /** Find the highest resolution subdirectory (Poly Haven style: 1K/, 2K/, 4K/).
    * Returns the subdirectory path, or the original directory if no res dirs found.
    */
  private def findHighestResDir(dir: Path): Path =
    val resPattern = """^(\d+)K$""".r
    Using(Files.list(dir)): stream =>
      val resDirs = stream
        .filter(Files.isDirectory(_))
        .toList.asScala.toList
        .flatMap: d =>
          d.getFileName.toString match
            case resPattern(k) => Some(k.toInt -> d)
            case _ => None
        .sortBy(-_._1) // Highest resolution first
      resDirs.headOption.map(_._2).getOrElse(dir)
    .getOrElse(dir)

  /** Resolve a texture set from a directory.
    *
    * @param dir       directory containing texture files
    * @param preferredRes optional resolution override (e.g., "2K")
    * @return ResolvedTextureSet with file paths for each detected map
    */
  def resolve(dir: Path, preferredRes: Option[String] = None): Try[ResolvedTextureSet] =
    if !Files.isDirectory(dir) then
      return Failure(new IllegalArgumentException(s"Not a directory: $dir"))

    // If Poly Haven style (res subdirs), pick the right resolution
    val searchDir = findHighestResDir(dir)

    val files = Using(Files.list(searchDir)): stream =>
      stream
        .filter(Files.isRegularFile(_))
        .filter(f => ImageExtensions.contains(extension(f).toLowerCase))
        .toList.asScala.toList
        .sortBy(f => -f.getFileName.toString.length)
    .getOrElse(Nil)

    if files.isEmpty then
      return Failure(new IllegalArgumentException(s"No image files found in $searchDir"))

    detectConvention(files) match
      case None =>
        Failure(new IllegalArgumentException(
          s"Could not detect PBR naming convention in $searchDir. " +
          "Expected ambientCG (*_Color.*, *_NormalGL.*) or Poly Haven (*_diff.*, *_nor_gl.*) patterns."))
      case Some(conv) =>
        val classified: Map[MapType, Path] = files.flatMap: f =>
          val name = f.getFileName.toString
          conv.classify(name) match
            case MapType.Unknown => None
            case mt => Some(mt -> f)
        .toMap

        val normalInfo = classified.get(MapType.NormalGL) match
          case Some(p) => (Some(p), false)
          case None => classified.get(MapType.NormalDX) match
            case Some(p) => (Some(p), true) // Needs DX→GL conversion
            case None => (None, false)

        Success(ResolvedTextureSet(
          color     = classified.get(MapType.Color),
          normal    = normalInfo._1,
          roughness = classified.get(MapType.Roughness),
          metallic  = classified.get(MapType.Metallic),
          ao        = classified.get(MapType.AO),
          height    = classified.get(MapType.Height),
          normalNeedsDXConversion = normalInfo._2
        ))

  private def extension(p: Path): String =
    val name = p.getFileName.toString
    val dot = name.lastIndexOf('.')
    if dot >= 0 then name.substring(dot) else ""
