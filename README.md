# Template description.

This template is based on [deeplearning4j word2vec example](https://github.com/deeplearning4j/dl4j-0.0.3.3-examples/tree/master/src/main/java/org/deeplearning4j/word2vec). It's goal is to show how to integrate deeplearning4j library with PredictionIO.

Word2Vec is algorithm used to computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research.

# Installation.

Follow [installation guide for PredictionIO](http://docs.prediction.io/install/).

After installation start all PredictionIO vendors and check pio status:
```bash
pio-start-all
pio status
```

This template depends on deeplearning4j 0.0.3.3.3.alpha1-SNAPSHOT. In order to install it run:
```bash
git clone git@github.com:deeplearning4j/deeplearning4j.git
cd deeplearning4j
chmod a+x setup.sh
./setup.sh
```

Copy this template to your local directory with:
```bash
pio template get ts335793/template-scala-spark-dl4j-word2vec <TemplateName>
```

# Build, train, deploy.

You might build template, train it and deploy by typing:
```bash
pio build
pio train -- --executor-memory=4GB --driver-memory=4GB
pio deploy -- --executor-memory=4GB --driver-memory=4GB
```
Those pio train options are used to avoid problems with java garbage collector. In case they appear increase executor memory and driver memory.

# Importing training data.

In this template there is are almost no restrictions for training data - each training example must consist of string of words.

You can import example training data from [kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data). It is collection of the Rotten Tomatoes movie reviews.

In order to use this data, create new app:
```bash
pio app new <ApplicationName> # prints out ApplicationAccessKey and ApplicationId
```
set appId in engine.json to ApplicationId and import data with:
```bash
python data/import_eventserver.py --access_key <ApplicationAccessKey> --file train.tsv
```

You can always remind your application id and key with:
```bash
pio app list
```

# Sending requests to server.

In order to send a query run in template directory:
```bash
python data/send_query_interactive.py
```
and type word. The result will be a list of nearest words.