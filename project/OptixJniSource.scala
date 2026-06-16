import java.io.File

import sbt.ProjectRef
import sbt.uri

object OptixJniSource {
  val commit = "7371676deca5e028c325c73186e27e427c5ee39d"
  val checkout: File = new File("../optix-jni").getCanonicalFile
  val projectRef: ProjectRef = ProjectRef(uri(checkout.toURI.toString), "optix-jni")
}
