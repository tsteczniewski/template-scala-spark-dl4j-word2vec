package org.template.vanilla

import io.prediction.controller._
import io.prediction.data.storage.Storage

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val eventsRDD: RDD[LabeledPhrase] = eventsDb
      .aggregateProperties(
        appId = dsp.appId,
        entityType = "phrase",
        required = Some(List("phrase")))(sc)
      .map({
        case (entityId, properties) =>
          LabeledPhrase(phrase = properties.get[String]("phrase"))
      })

    new TrainingData(eventsRDD)
  }
}

case class LabeledPhrase(
  phrase: String
)

class TrainingData(
  val labeledPhrases: RDD[LabeledPhrase]
) extends Serializable with SanityCheck {
  override def toString = {
    s"events: [${labeledPhrases.count()}] (${labeledPhrases.take(2).toList}...)"
  }

  override def sanityCheck(): Unit = {
    assert(labeledPhrases.count > 0)
  }
}
