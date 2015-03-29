package org.template.vanilla

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(word: String) extends Serializable

case class PredictedResult(nearestWords: List[String]) extends Serializable

object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("nearestWords" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
