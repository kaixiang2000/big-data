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


#8 Find out the top 3 airline carriers that have the largest departure delay for each day of the week.
flights.groupBy("AIRLINE",'DAY_OF_WEEK').agg(f.sum("DEPARTURE_DELAY").alias('MAX_DEPARTURE_DELAY')).orderBy("MAX_DEPARTURE_DELAY", ascending=False)\
    .limit(3).show()
sc.stop()