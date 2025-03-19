package menger.objects.higher_d

case class FixedVector[n <: Int & Singleton, T](values: T*)(implicit d: ValueOf[n]):

  require(values.size == d.value, s"Expected ${d.value} elements, got ${values.size}")

  def dimension: Int = d.value

  def apply(i: Int): T =
    require(i >= 0 && i < d.value, s"Index $i out of bounds")
    values(i)

  def toList: List[T] = values.toList
