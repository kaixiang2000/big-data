import findspark
findspark.init()
from pyspark import SparkConf,SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as f

spark=SparkSession.builder.getOrCreate()
sc=SparkContext.getOrCreate()
sqlContext=SQLContext(sc)
flights=sqlContext.read.format('csv').options(header='true', inferSchema='true', sep=",")\
                  .load(r"C:/Users/yanqi/OneDrive/Desktop/flights-small.csv")



flights.filter('DISTANCE>=2500').withColumn('count',f.lit(1)).groupBy('AIRLINE')\
    .agg(f.sum("count").alias("longest")).orderBy("longest",ascending=False).limit(3).show()


sc.stop()