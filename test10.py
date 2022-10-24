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



#10.
df=flights.select('ORIGIN_AIRPORT','DESTINATION_AIRPORT')
df1= df.alias("l").join(df.alias("r"),
f.col("l.DESTINATION_AIRPORT") == f.col("r.ORIGIN_AIRPORT"), "left").select(*["l.ORIGIN_AIRPORT", "r.DESTINATION_AIRPORT"])
df1.filter(df1.ORIGIN_AIRPORT=='LAX').filter(df1.DESTINATION_AIRPORT=='JFK').show()

sc.stop()