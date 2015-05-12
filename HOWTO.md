# Integrating Deeplearnign4j library

This file presents steps that must be taken in order to change PredictionIO vanilla template to this template and integrate Deeplearning4j library.

## Install Deeplearning4j 0.0.3.3.3.alpha1-SNAPSHOT

```bash
git clone git@github.com:deeplearning4j/deeplearning4j.git
cd deeplearning4j
chmod a+x setup.sh
./setup.sh
```

## Modify build.sbt

In order to use Deeplearning4j in your template you must add it along with libraries it depends on to dependencies in your build.sbt.
```scala
libraryDependencies ++= Seq(
  "io.prediction"      %% "core"                % pioVersion.value   % "provided",
  "org.apache.spark"   %% "spark-core"          % "1.3.0"   % "provided",
  "org.apache.spark"   %% "spark-mllib"         % "1.3.0"   % "provided",
  "org.deeplearning4j" %  "deeplearning4j-core" % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.deeplearning4j" %  "deeplearning4j-nlp"  % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.deeplearning4j" %  "dl4j-spark"          % "0.0.3.3.3.alpha1-SNAPSHOT" 
    exclude("org.slf4j", "slf4j-api"), // ADDED
  "org.deeplearning4j" %  "dl4j-spark-nlp"      % "0.0.3.3.3.alpha1-SNAPSHOT", // ADDED
  "org.nd4j"           %  "nd4j-jblas"          % "0.0.3.5.5.3-SNAPSHOT", //ADDED
  "com.google.guava"   %  "guava"               % "14.0.1" // ADDED
)
```

In order to handle dependencies conflicts when installing deeplearning4j add this merge strategy to build.sbt.
```scala
// ADDED
mergeStrategy in assembly := {
  case x if Assembly.isConfigFile(x) =>
    MergeStrategy.concat
  case PathList(ps @ _*) if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) =>
    MergeStrategy.rename
  case PathList("META-INF", xs @ _*) =>
    (xs map {_.toLowerCase}) match {
      case ("manifest.mf" :: Nil) | ("index.list" :: Nil) | ("dependencies" :: Nil) =>
        MergeStrategy.discard
      case ps@(x :: xs) if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
        MergeStrategy.discard
      case "plexus" :: xs =>
        MergeStrategy.discard
      case "services" :: xs =>
        MergeStrategy.filterDistinctLines
      case ("spring.schemas" :: Nil) | ("spring.handlers" :: Nil) =>
        MergeStrategy.filterDistinctLines
      case _ => MergeStrategy.first
    }
  case PathList(_*) => MergeStrategy.first
}
```

## Update Engine.scala

Update Query to include word, and the Predict Result to include list of words.
```scala
case class Query(word: String) extends Serializable // CHANGED

case class PredictedResult(nearestWords: List[String]) extends Serializable // CHANGED
```

Change algorithm name.
```scala
object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("nearestWords" -> classOf[Algorithm]), // CHANGED
      classOf[Serving])
  }
}
```

## Update engine.json

Deeplearning4j spark Word2Vec requires setting "negative" option to zero in spark config.
```scala
{
  "id": "default",
  "description": "Default settings",
  "engineFactory": "org.template.vanilla.VanillaEngine",
  // ADDED
  "sparkConf": {
    "org.deeplearning4j.scaleout.perform.models.word2vec": {
      "negative": 0
    }
  },
  "datasource": {
    "params" : {
      "appName": "MyAppName"
    }
  },
  "algorithms": [
    {
      "name": "nearestWords", // CHANGED
      "params": {
        "nearestWordsQuantity": 10 // CHANGED
      }
    }
  ]
}
```

## Update import_eventserver.py

Adjust data/import_eventserver.py to Kaggle's data.

## Update DataSource.scala

Create Labeled Phrase class representing one phrase from data set and modify Training Data to return list of Labeled Phrases. Add sanity check that will be runned each time training will be executed.
```scala
// ADDED
case class LabeledPhrase(
  phraseId: Int,
  sentenceId: Int,
  phrase: String,
  sentiment: Int
)

class TrainingData(
  val labeledPhrases: RDD[LabeledPhrase] // CHANGED
) extends Serializable with SanityCheck { // CHANGED
  // CHANGED
  override def toString = {
    s"events: [${labeledPhrases.count()}] (${labeledPhrases.take(2).toList}...)"
  }

  // ADDED
  override def sanityCheck(): Unit = {
    assert(labeledPhrases.count > 0)
  }
}
```

Adjust readTrainging to load Kaggle's data from database.
```scala
class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]
  
  // CHANGED
  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsRDD: RDD[LabeledPhrase] = PEventStore
      .aggregateProperties(
        appName = dsp.appName,
        entityType = "phrase",
        required = Some(List("sentenceId", "phrase", "sentiment")))(sc)
      .map({
        case (entityId, properties) =>
          LabeledPhrase(
            phraseId = entityId.toInt,
            sentenceId = properties.get[String]("sentenceId").toInt,
            phrase = properties.get[String]("phrase"),
            sentiment = properties.get[String]("sentiment").toInt
          )
      })

    new TrainingData(eventsRDD)
  }
}
```

## Update Preparator.scala

Prepare Kaggle's data from Data Source by mapping it to list of phrases.  
```scala
class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(phrases = trainingData.labeledPhrases.map { _.phrase }) // CHANGED
  }
}
```

## Modify Algorithm.scala

Include Deeplearning4j libraries.
```scala
import org.deeplearning4j.models.embeddings.WeightLookupTable // ADDED
import org.deeplearning4j.models.word2vec.Word2Vec // ADDED
import org.deeplearning4j.models.word2vec.wordstore.VocabCache // ADDED
import org.deeplearning4j.spark.models.embeddings.word2vec.{Word2Vec => SparkWord2Vec} // ADDED
```

Change AlgorithmParams to store information about number of nearest words returned in query result.
```scala
case class AlgorithmParams(nearestWordsQuantity: Integer) extends Params // CHANGED
```

Make Model store Word2Vec model.
```scala
// CHANGED
class Model(
  val vocabCache: VocabCache,
  val weightLookupTable: WeightLookupTable
) extends Serializable
```

Modify Algorithm. Changes are:
* train function runs Word2Vec algorithm's training on phrases from Data Preparator and returns new model,
* predict function returns nearest words to the word given in query.
```scala
class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]
  
  // CHANGED
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

  // CHANGED
  def predict(model: Model, query: Query): PredictedResult = {
    val word2Vec = new Word2Vec.Builder()
      .vocabCache(model.vocabCache)
      .lookupTable(model.weightLookupTable)
      .build()
    val nearestWords = word2Vec.wordsNearest(query.word, ap.nearestWordsQuantity)
    PredictedResult(nearestWords = nearestWords.toList)
  }
}
```