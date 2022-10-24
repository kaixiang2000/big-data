import findspark
findspark.init()
from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

spark=SparkSession.builder.getOrCreate()
sc=SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
train=sqlContext.read.format('csv').options(header='true', inferSchema='true', sep=",").load(r"C:/Users/yanqi/OneDrive/Desktop/wine-train.csv")
test=sqlContext.read.format('csv').options(header='true', inferSchema='true', sep=",").load(r"C:/Users/yanqi/OneDrive/Desktop/wine-test.csv")
train=train.toDF(*(c.replace(' ', '_') for c in train.columns))
test=test.toDF(*(c.replace(' ', '_') for c in test.columns))
train.groupBy("label").count().show()
p_weight = train.filter('label == 1').count()/ train.count()
n_weight = train.filter('label == 0').count()/ train.count()
print(n_weight, p_weight)
train = train.withColumn("weight", F.when(F.col("label")==1,n_weight).otherwise(p_weight))
feature_columns = ["fixed_acidity","volatile_acidity",'citric_acid',"residual_sugar",'chlorides','free_sulfur_dioxide','total_sulfur_dioxide'
                   ,'density','pH','sulphates','sulphates','alcohol','red']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
pipeline_p = Pipeline(stages=[assembler])
data_model = pipeline_p.fit(train)
transformed_data = data_model.transform(train)
transformed_data.show(5)
transformed_test = data_model.transform(test)
transformed_test.show(5)
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

#logistic
cassifier = LogisticRegression(maxIter=5, featuresCol = "features", weightCol="weight")
pipeline = Pipeline(stages=[cassifier])
model = pipeline.fit(transformed_data)
m_metrics_l(model,transformed_test)

#svm
cassifier = LinearSVC(maxIter=10,  weightCol="weight")
pipeline = Pipeline(stages=[cassifier])
model = pipeline.fit(transformed_data)
m_metrics_l(model,transformed_test)

#improvement
cassifier = GBTClassifier(maxIter=50, featuresCol = "features", weightCol="weight")
pipeline = Pipeline(stages=[cassifier])
model = pipeline.fit(transformed_data)
m_metrics_l(model,transformed_test)