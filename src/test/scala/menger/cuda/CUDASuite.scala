package menger.cuda

import org.scalatest.flatspec.AnyFlatSpec
import jcuda.{JCudaVersion, Pointer}
import jcuda.runtime.JCuda

class CUDASuite extends AnyFlatSpec:
  def isCudaAvailable: Boolean = JCudaVersion.isAvailable()

  "JCUDA" should "be able to be created" in:
    assume(isCudaAvailable, "CUDA not available")
    val pointer = Pointer()
    JCuda.cudaMalloc(pointer, 4)
    JCuda.cudaFree(pointer)
