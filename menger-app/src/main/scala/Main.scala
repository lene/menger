
import java.nio.file.Paths

import scala.jdk.CollectionConverters._
import scala.jdk.OptionConverters._

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import com.badlogic.gdx.ApplicationListener
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import menger.GlobalRotation
import menger.MengerCLIOptions
import menger.MengerExitException
import menger.cli.LightSpec
import menger.common.Const
import menger.common.FogConfig
import menger.common.ProfilingConfig
import menger.common.RenderConfig
import menger.config.CameraConfig
import menger.config.EnvironmentConfig
import menger.config.ExecutionConfig
import menger.config.OptiXEngineConfig
import menger.config.PlaneConfig
import menger.config.SceneConfig
import menger.config.TAnimationConfig
import menger.dsl.DenoiseMode
import menger.dsl.LoadedScene
import menger.dsl.RestrictedClasspath
import menger.engines.AnimationEngine
import menger.engines.CliAnimationEngine
import menger.engines.InteractiveEngine
import menger.engines.PreviewEngine
import menger.engines.RenderEngine
import menger.engines.RenderLock
import menger.engines.SceneConverter
import menger.engines.VideoEngine
import menger.tools.SceneValidator
import org.slf4j.LoggerFactory
import upickle.default.write

object Main:
  def main(args: Array[String]): Unit =
    try
      val opts = MengerCLIOptions(args.toList)
      configureLogging(opts.logLevel().toUpperCase)
      opts.display.toOption match
        // AD-6: the display target is an explicit, injected parameter. A native windowing
        // library (GLFW/X11, via LWJGL) only ever reads DISPLAY from the process environment
        // at init time -- nothing in-process can override it once the JVM has started, so the
        // only way to honor an injected value is to re-exec as a child process that has it
        // set from the start. When --display is absent this branch is never taken, so every
        // existing invocation keeps today's in-process, ambient-environment behavior exactly.
        case Some(display) if display.trim.nonEmpty => sys.exit(reExecWithDisplay(args, display))
        case Some(_) => sys.error("--display was given an empty value")
        case None => launchInProcess(opts)
    catch
      case e: MengerExitException => sys.exit(e.code)
      case e: Exception =>
        System.err.println(s"Error: ${e.getMessage}")
        sys.exit(1)

  /** AD-16: the GPU is a single exclusive resource. Only the genuine interactive window --
    * a real-time session an `InteractiveEngine` drives for however long the user keeps it
    * open -- is gated; headless/preview/video/animation runs are one-shot batch renders
    * already sequential-only by existing convention (AD-11), and this story's own
    * Boundaries explicitly excludes them from locking. A pure predicate, kept separate from
    * `launchInProcess`'s side-effecting match, so the discrimination itself (not just that
    * the code compiles) is directly unit-testable without touching `Lwjgl3Application`. */
  def shouldLock(rendering: RenderEngine, opts: MengerCLIOptions): Boolean =
    rendering match
      case _: InteractiveEngine => !opts.headless()
      case _ => false

  private def launchInProcess(opts: MengerCLIOptions): Unit =
    val config = getConfig(opts)
    val rendering = createEngine(opts)
    rendering match
      case app: ApplicationListener if shouldLock(rendering, opts) =>
        RenderLock.tryAcquire(opts.renderLockPath()) match
          case Right(lock) =>
            try Lwjgl3Application(app, config)
            finally lock.close()
          case Left(reason) => reportRefusedAndExit(reason)
      case app: ApplicationListener => Lwjgl3Application(app, config)
      case _ => sys.error("Engine must implement ApplicationListener")

  /** Reuses `SceneValidator`'s AD-5 tagged-result JSON shape rather than inventing a second
    * "refused" contract -- one tagged-result vocabulary across the validation gauntlet and
    * the render-exclusivity check. Pure JSON construction, kept separate from `sys.exit` so
    * the contract itself is directly testable (mirroring `buildReExecProcessBuilder`'s own
    * separation from `reExecWithDisplay`). `Main` already matches `ArchitectureSpec`'s
    * `.*Main.*` exemption for stdout/`sys.exit`, so no separate isolation object is needed. */
  def refusedResultJson(reason: String): String =
    write(SceneValidator.ValidationResult(SceneValidator.Tag.Refused, List(reason)), indent = 2)

  private def reportRefusedAndExit(reason: String): Unit =
    print(refusedResultJson(reason))
    print(System.lineSeparator())
    sys.exit(1)

  /** Pure command/environment construction, kept separate from `reExecWithDisplay` so tests
    * can assert the child's environment carries the injected `DISPLAY` value without
    * actually starting a process (this environment has no real display to render into).
    * Strips every `--display`/`--display=value` occurrence from `rawArgs` before forwarding
    * them to the child (review round: stripping only the first occurrence left a second one
    * in place, which would re-trigger this same re-exec branch in the child and recurse) --
    * otherwise the child would see `--display` again and re-exec itself forever.
    *
    * Classpath: `RestrictedClasspath.fullClasspath` (story 5), not a bare
    * `System.getProperty("java.class.path")` (review round) -- that module's own doc comment
    * already established why the system property alone is incomplete under sbt's layered
    * classloaders ("neither alone is complete"); reusing the existing, already-correct
    * enumeration here rather than re-deriving a narrower one avoids reintroducing the same
    * gap in a second place. */
  def buildReExecProcessBuilder(rawArgs: Array[String], display: String): ProcessBuilder =
    val javaBin = Paths.get(System.getProperty("java.home"), "bin", "java").toString
    val classpath = RestrictedClasspath.fullClasspath
    val childArgs = stripDisplayFlag(rawArgs)
    val command = (List(javaBin, "-cp", classpath, "Main") ++ childArgs.toList).asJava
    val builder = ProcessBuilder(command)
    builder.environment().put("DISPLAY", display)
    builder.inheritIO()
    builder

  private def stripDisplayFlag(args: Array[String]): Array[String] =
    val withoutValues = args.indices.filterNot { i =>
      args(i) == "--display" || (i > 0 && args(i - 1) == "--display")
    }
    withoutValues.map(args).filterNot(_.startsWith("--display=")).toArray

  /** Ties the child's lifetime to the parent's: without this, a parent killed abruptly
    * (SIGTERM/SIGKILL, a supervisor terminating it) while blocked in `waitFor()` (review
    * round) would leave the child running detached -- orphaned, and still holding the
    * render lock the whole point of this re-exec is to eventually pass through to. */
  private def reExecWithDisplay(rawArgs: Array[String], display: String): Int =
    val process = buildReExecProcessBuilder(rawArgs, display).start()
    val shutdownHook = new Thread(() => if process.isAlive then process.destroy())
    Runtime.getRuntime.addShutdownHook(shutdownHook)
    try process.waitFor()
    finally
      try Runtime.getRuntime.removeShutdownHook(shutdownHook)
      catch case _: IllegalStateException => () // already shutting down -- hook will run anyway

  private def configureLogging(levelName: String): Unit =
    val level = Level.valueOf(levelName)
    // SLF4J returns the interface, but we need Logback's implementation to set the level
    LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME) match
      case logger: Logger =>
        logger.setLevel(level)
      case other =>
        val className = other.getClass.getName
        System.err.println(s"Warning: Expected Logback Logger but got $className")
        System.err.println("Logging level not set.")

  def getConfig(opts: MengerCLIOptions): Lwjgl3ApplicationConfiguration =
    val config = Lwjgl3ApplicationConfiguration()
    config.disableAudio(true)
    config.setTitle("Menger Sponges")
    config.setWindowedMode(opts.width(), opts.height())
    config.setBackBufferConfig(
      Const.Display.colorBits, Const.Display.colorBits,
      Const.Display.colorBits, Const.Display.colorBits,
      Const.Display.depthBits, Const.Display.stencilBits,
      opts.antialiasSamples()
    )
    // OptiX rendering requires fixed resolution for optimal performance
    config.setResizable(false)
    // Headless mode: render without displaying window (for CI/CD, batch processing)
    if opts.headless() then
      config.setInitialVisible(false)
      config.setDecorated(false)
    config

  def createEngine(opts: MengerCLIOptions): RenderEngine =
    given ProfilingConfig = opts.profileMinMs.toOption match
      case Some(minMs) => ProfilingConfig.enabled(minMs)
      case None => ProfilingConfig.disabled

    opts.scene.toOption match
      case Some(sceneName) => createSceneBasedEngine(opts, sceneName)
      case None => createCliBasedOptiXEngine(opts)

  private def createSceneBasedEngine(opts: MengerCLIOptions, sceneName: String)(using ProfilingConfig): RenderEngine =
    // Ensure all example scene objects are initialized so short names are registered
    val _ = examples.dsl.SceneIndex

    import menger.dsl.SceneLoader
    SceneLoader.load(sceneName) match
      case Right(LoadedScene.Animated(fn)) if opts.preview() =>
        val animConfig = TAnimationConfig(
          startT      = opts.startT(),
          endT        = opts.endT(),
          frames      = opts.tFrames.toOption.getOrElse(100),
          savePattern = ""
        )
        PreviewEngine(
          sceneFunction   = fn,
          previewConfig   = animConfig,
          executionConfig = buildExecutionConfig(opts),
          renderConfig    = opts.renderConfig,
          causticsConfig  = opts.causticsConfig,
          denoiseModeOverride = cliDenoiseOverride(opts)
        )

      case Right(LoadedScene.Animated(fn)) if opts.tFrames.isSupplied =>
        // Multi-frame animation: create VideoEngine (with ffmpeg) or AnimationEngine (frames only)
        val animConfig = TAnimationConfig(
          startT = opts.startT(),
          endT = opts.endT(),
          frames = opts.tFrames(),
          savePattern = opts.saveName()
        )
        if opts.video.isSupplied then
          VideoEngine(
            sceneFunction = fn,
            animConfig = animConfig,
            executionConfig = buildExecutionConfig(opts),
            renderConfig = opts.renderConfig,
            causticsConfig = opts.causticsConfig,
            videoOutputPath = opts.video(),
            videoQuality = opts.videoQuality(),
            keepFrames = opts.keepFrames(),
            denoiseModeOverride = cliDenoiseOverride(opts)
          )
        else
          AnimationEngine(
            sceneFunction = fn,
            animConfig = animConfig,
            executionConfig = buildExecutionConfig(opts),
            renderConfig = opts.renderConfig,
            causticsConfig = opts.causticsConfig,
            denoiseModeOverride = cliDenoiseOverride(opts)
          )

      case Right(loadedScene) =>
        // Static scene or animated scene evaluated at fixed t
        val freezeT = opts.freezeT.toOption.getOrElse(0f)
        val dslScene = loadedScene match
          case LoadedScene.Static(scene) => scene
          case LoadedScene.Animated(fn) => fn(freezeT)
        createOptiXEngineFromDslScene(opts, dslScene, freezeT)

      case Left(error) =>
        System.err.println(s"Failed to load scene '$sceneName': $error")
        sys.exit(1)

  private def createOptiXEngineFromDslScene(
    opts: MengerCLIOptions, dslScene: menger.dsl.Scene, renderT: Float = 0f
  )(using ProfilingConfig): InteractiveEngine =
    val configs = SceneConverter.convert(dslScene, opts.causticsConfig)
    val baseRender = configs.render.getOrElse(RenderConfig.Default)
    val mergedRender = RenderConfig(
      shadows            = if opts.shadows.isSupplied            then opts.shadows()            else baseRender.shadows,
      transparentShadows = if opts.transparentShadows.isSupplied then opts.transparentShadows() else baseRender.transparentShadows,
      antialiasing       = if opts.antialiasing.isSupplied       then opts.antialiasing()       else baseRender.antialiasing,
      aaMaxDepth         = if opts.aaMaxDepth.isSupplied         then opts.aaMaxDepth()         else baseRender.aaMaxDepth,
      aaThreshold        = if opts.aaThreshold.isSupplied        then opts.aaThreshold()        else baseRender.aaThreshold,
      toneMappingOperator = configs.toneMappingOperator,
      toneMappingExposure = configs.toneMappingExposure,
    )
    val mergedDenoise = cliDenoiseOverride(opts).getOrElse(configs.denoiseMode)
    val mergedAccumulation =
      if opts.accumulationFrames.isSupplied then opts.accumulationFrames()
      else configs.accumulationFrames
    val engineConfig = OptiXEngineConfig(
      scene = configs.scene,
      camera = configs.camera,
      environment = EnvironmentConfig(
        planes = configs.planes,
        lights = configs.lights,
        background = configs.background,
        fog = configs.fog,
        envMap = configs.envMap,
        envMapVideo = configs.envMapVideo,
        iblEnabled = configs.iblEnabled,
        iblStrength = configs.iblStrength,
        iblSamples = configs.iblSamples
      ),
      execution = buildExecutionConfig(opts),
      render = mergedRender,
      caustics = configs.caustics,
      cross = opts.crossConfig,
      denoiseMode = mergedDenoise,
      accumulationFrames = mergedAccumulation
    )
    InteractiveEngine(engineConfig, opts.userSetMaxInstances, renderT)

  private def createCliBasedOptiXEngine(opts: MengerCLIOptions)(using ProfilingConfig): RenderEngine =
    // S2 menger#33: these three flags are validated (CliValidation's mutual-exclusion check)
    // but never applied anywhere below -- --objects type=...:color=#RRGGBB is the real,
    // wired mechanism. Not removed (that broke CliValidation's coupling when tried); warn
    // instead so the silence stops.
    if opts.color.isSupplied || opts.faceColor.isSupplied || opts.lineColor.isSupplied then
      LoggerFactory.getLogger("Main").warn(
        "--color/--face-color/--line-color have no effect on rendering -- use " +
        "--objects type=...:color=#RRGGBB instead"
      )
    val engineConfig = OptiXEngineConfig(
      scene = SceneConfig(objectSpecs = opts.objects.toOption.map(GlobalRotation(opts, _))),
      camera = CameraConfig(
        position = opts.cameraPos(),
        lookAt = opts.cameraLookat(),
        up = opts.cameraUp()
      ),
      environment = EnvironmentConfig(
        planes = opts.plane.toOption.toList.map(p => PlaneConfig(
          p,
          opts.planeColor.toOption,
          opts.planeMaterial.toOption.flatMap(s => menger.common.Material.fromName(s).toScala)
        )),
        lights = opts.light.toOption.getOrElse(List.empty).map(LightSpec.toCommonLight),
        envMap = opts.envMap.toOption,
        fog = opts.fog.toOption.map(f => FogConfig(f.density, f.color))
      ),
      execution = buildExecutionConfig(opts),
      render = opts.renderConfig,
      caustics = opts.causticsConfig,
      cross = opts.crossConfig,
      denoiseMode = opts.denoiseMode,
      accumulationFrames = opts.accumulationFrames()
    )
    opts.animate.toOption match
      case Some(animSpec) =>
        CliAnimationEngine(engineConfig, animSpec, opts.saveName.toOption)
      case None =>
        InteractiveEngine(engineConfig, opts.userSetMaxInstances)

  private def buildExecutionConfig(opts: MengerCLIOptions): ExecutionConfig =
    ExecutionConfig(
      fpsLogIntervalMs = opts.fpsLogInterval(),
      timeout = opts.timeout(),
      saveName = opts.saveName.toOption,
      enableStats = opts.stats() || opts.headless() || opts.statsJson.isSupplied,
      maxInstances = opts.maxInstances(),
      textureDir = opts.textureDir(),
      allowUniformRender = opts.allowUniformRender(),
      statsJsonPath = opts.statsJson.toOption
    )

  private def cliDenoiseOverride(opts: MengerCLIOptions): Option[DenoiseMode] =
    if opts.denoiseModeSupplied then Some(opts.denoiseMode) else None
