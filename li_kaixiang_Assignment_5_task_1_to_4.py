from __future__ import print_function
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import PCA
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
import time
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

# If needed, use this helper function
# You can implement your own version if you find it more appropriate 

if __name__ == "__main__":
	# Use this code to read the data
    ##TRAIN
    # Use this code to reade the data
    corpus = sc.textFile(sys.argv[1], 1)
    keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    df = spark.createDataFrame(keyAndText).toDF("id","label","text").cache()

    ##TEST
    #Use this code to read the data
    corpust = sc.textFile(sys.argv[2], 1)
    keyAndTextt = corpust.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])).map(lambda x: (x[0], int(x[0].startswith("AU")),x[1]))   
    # Spark DataFrame to be used wiht MLlib 
    test = spark.createDataFrame(keyAndTextt).toDF("id","label","text").cache()

    start = time.time()
    print("start counting time for task1")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
    countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="rawFeatures", vocabSize=5000)
    idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="featuresIDF")
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf])
    # Train the model
    data_model = pipeline_p.fit(df)
    print(f" Total time needed to vectorize the data is {time.time()-start:.2f}s.")

    # Get the vocabulary of the CountVectroizer
    print("First ten words of the vocabulary: \n",data_model.stages[2].vocabulary[:10])
    ### Task 2
    ### Build your learning model using Logistic Regression
    p_weight = df.filter('label == 1').count()/ df.count()
    n_weight = df.filter('label == 0').count()/ df.count()
    df = df.withColumn("weight", F.when(F.col("label")==1,n_weight).otherwise(p_weight))
    transformed_data = data_model.transform(df)
    transformed_test = data_model.transform(test)
    print("\nPerformance Metrics: Logisitic Regression")
    def m_metrics_l(ml_model,test_data):
        predictions = ml_model.transform(test_data).cache()
        predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
        
        # Print some predictions vs labels
        # print(predictionAndLabels.take(10))
        metrics = MulticlassMetrics(predictionAndLabels)
        
        # Overall statistics
        precision = metrics.precision(1.0)
        recall = metrics.recall(1.0)
        f1Score = metrics.fMeasure(1.0)
        print(f"Precision = {precision:.4f} Recall = {recall:.4f} F1 Score = {f1Score:.4f}")
        print("Confusion matrix \n", metrics.confusionMatrix().toArray().astype(int))
    print("\nPerformance Metrics: LogisticRegression")
    cassifier = LogisticRegression(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    start = time.time()
    pipeline = Pipeline(stages=[cassifier])
    print(f"Training started.")
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")
    ### Task 3
    ### Build your learning model using SVM
    print("\nPerformance Metrics: SVM")
    cassifier = LinearSVC(maxIter=20, featuresCol = "featuresIDF", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    print(f"Training started.")
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")
    ### Task 4
    print("task4 start")
    selector = ChiSqSelector(numTopFeatures=200, featuresCol=idf.getOutputCol(), outputCol="features", labelCol="label")
    pipeline_p = Pipeline(stages=[tokenizer,remover, countVectorizer, idf,selector])
    data_model = pipeline_p.fit(df)
    transformed_data = data_model.transform(df)
    transformed_test = data_model.transform(test)
    print("task4 restart LogisticRegression")
    cassifier = LogisticRegression(maxIter=20, featuresCol = "features", weightCol="weight")
    start = time.time()
    pipeline = Pipeline(stages=[cassifier])
    print(f"Training started.")
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")
    print("task4 restart Llinear svc")
    cassifier = LinearSVC(maxIter=20, featuresCol = "features", weightCol="weight")
    pipeline = Pipeline(stages=[cassifier])
    start = time.time()
    print(f"Training started.")
    model = pipeline.fit(transformed_data)
    print(f"Model created in {time.time()-start:.2f}s.")
    m_metrics_l(model,transformed_test)
    print(f"Total time {time.time()-start:.2f}s.")
    sc.stop()

    
