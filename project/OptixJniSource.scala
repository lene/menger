import java.io.File

import sbt.ProjectRef
import sbt.uri

object OptixJniSource {
  val commit = "7371676deca5e028c325c73186e27e427c5ee39d"
  val checkout: File = new File("../optix-jni").getCanonicalFile
  val usesGitSource: Boolean =
    sys.env.get("MENGER_OPTIX_JNI_SOURCE").exists(_.equalsIgnoreCase("git"))
  val usesLocalCheckout: Boolean = !usesGitSource && new File(checkout, "build.sbt").isFile
  val sourceUri: String =
    if (usesLocalCheckout) {
      checkout.toURI.toString
    } else {
      s"https://github.com/lene/optix-jni.git#$commit"
    }
  val projectRef: ProjectRef = ProjectRef(uri(sourceUri), "optix-jni")
}
