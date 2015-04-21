package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.SparkContext

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.spark.models.embeddings.word2vec.{Word2Vec => SparkWord2Vec}

import scala.collection.JavaConversions._

import grizzled.slf4j.Logger

case class AlgorithmParams(nearestWordsQuantity: Integer) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val (vocabCache, weightLookupTable) = {
      val result = new SparkWord2Vec().train(data.phrases)
      (result.getFirst, result.getSecond)
    }
    new Model(
      vocabCache = vocabCache,
      weightLookupTable = weightLookupTable
    )
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val word2Vec = new Word2Vec.Builder()
      .vocabCache(model.vocabCache)
      .lookupTable(model.weightLookupTable)
      .build()
    val nearestWords = word2Vec.wordsNearest(query.word, ap.nearestWordsQuantity)
    PredictedResult(nearestWords = nearestWords.toList)
  }
}

class Model(
  val vocabCache: VocabCache,
  val weightLookupTable: WeightLookupTable
) extends Serializable
