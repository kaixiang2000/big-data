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


#Which day of the week has the largest flight cancelation ratio?
flights.withColumn("TOTAL", f.lit(1)).groupBy('DAY_OF_WEEK')\
    .agg(f.sum("CANCELLED").alias("CANCELLED"), f.sum("TOTAL").alias("TOTAL"))\
        .withColumn('ratio',f.col('CANCELLED')/f.col("Total")).orderBy('ratio',ascending=False).limit(1).show()


sc.stop()