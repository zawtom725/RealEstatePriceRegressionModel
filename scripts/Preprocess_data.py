import pyspark.sql as sql
import pyspark.sql.functions as F
import pyspark.ml.feature as feature
import sys

spark = sql.SparkSession.builder.getOrCreate()

TRAIN_FILE = 'hdfs:///data/properties_2016.csv'
TEST_FILE = 'hdfs:///data/properties_2017.csv'

test_df = spark.read.csv(TEST_FILE, header=True)
data_df = spark.read.csv(TRAIN_FILE, header=True)

# label_df.show()
# data_df.show()

# new_df = label_df.join(data_df, label_df.parcelid == data_df.parcelid).drop(data_df.parcelid)
# new_df.show()

# COLUMN = ['airconditioningtypeid',
#           'fips',
#           'bathroomcnt',
#           'bedroomcnt',
#           'buildingqualitytypeid',
#           'calculatedbathnbr',
#           'calculatedfinishedsquarefeet',
#           'regionidzip',
#           'roomcnt',
#           'yearbuilt',
#           'taxamount',
#           'taxvaluedollarcnt']

COLUMN = ['airconditioningtypeid',
          'fips',
          'bathroomcnt',
          'lotsizesquarefeet',
          'bedroomcnt',
          'buildingqualitytypeid',
          'calculatedbathnbr',
          'calculatedfinishedsquarefeet',
          'regionidzip',
          'roomcnt',
          'yearbuilt',
          'taxamount',
          'taxvaluedollarcnt']

selected_df = data_df.select(*COLUMN) \
    .withColumn('airconditioningtypeid', F.when(
    data_df.airconditioningtypeid != None,
    data_df.airconditioningtypeid).otherwise(
    5) - 1).dropna()
# normalize to 0-index
selected_df.show()
selected_df.write.format('csv').mode('overwrite').save('gs://property_nn/train')

# test data
selected_test_df = test_df.select(*COLUMN) \
    .withColumn('airconditioningtypeid', F.when(
    test_df.airconditioningtypeid != None,
    test_df.airconditioningtypeid).otherwise(
    5) - 1).dropna()
selected_test_df.write.format('csv').mode('overwrite').save('gs://property_nn/test')
