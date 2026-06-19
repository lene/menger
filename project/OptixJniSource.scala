import java.io.File

import sbt.ProjectRef
import sbt.uri

object OptixJniSource {
  val commit = "7cfabdd7e5df7362d226d5e0ec59267422c98454"
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
