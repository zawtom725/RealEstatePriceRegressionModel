from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, RidgeRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark import SparkContext, SparkConf
import argparse
import csv
import re

def tql_regression(sc, filename):
    # load in csv
    records = sc.textFile(filename).sample(False, 0.1, 16)
    p_records = records.flatMap(lambda l: process_record(l))

    prices = p_records.keys()
    attrs = p_records.values()

    labeled_points = prices.zip(attrs).map(lambda x: LabeledPoint(x[0], x[1]))
    print(labeled_points.take(10))

    training, test = labeled_points.randomSplit([0.8, 0.2])
    #model = RidgeRegressionWithSGD.train(training, iterations=200)
    model = LinearRegressionWithSGD.train(training, iterations=200, regType="l2")

    # Use our model to predict
    train_predicts = model.predict(training.map(lambda x: x.features))
    train_preds = training.map(lambda x: x.label).zip(train_predicts)
    test_preds = test.map(lambda x: x.label).zip(model.predict(test.map(lambda x: x.features)))

    print(train_preds.take(10))
    print(test_preds.take(10))

    # Ask PySpark for some metrics on how our model predictions performed
    trained_metrics = RegressionMetrics(train_preds.map(lambda x: ( x[0], float(x[1]) )))
    test_metrics = RegressionMetrics(test_preds.map(lambda x: ( x[0], float(x[1]) )))

    print("___________________trained RMSE", trained_metrics.rootMeanSquaredError)
    print("___________________trained EV", trained_metrics.explainedVariance)

    print("___________________test RMSE", test_metrics.rootMeanSquaredError)
    print("___________________test EV", test_metrics.explainedVariance)

    return 0

# helper functions
def process_record(line):
    revw = csv.reader([str(line)])
    l = list(revw)[0]
    if l[0] != 'parcelid' and len(l) == 58:
        # the attributes to regress on
        vec = []
        basementsqft = l[3]
        vec.append(basementsqft)
        bathrmcnt = l[4]
        vec.append(bathrmcnt)
        bdrmcnt = l[5]
        vec.append(bdrmcnt)
        calcbathrmcnt = l[8]
        vec.append(calcbathrmcnt)
        floorsqft = l[10]
        vec.append(floorsqft)
        calcfloorsqft = l[11]
        vec.append(calcfloorsqft)
        fullbathrmcnt = l[19]
        vec.append(fullbathrmcnt)
        garagesqft = l[21]
        vec.append(garagesqft)
        lotsizesqft = l[26]
        vec.append(lotsizesqft)
        poolcnt = l[27]
        vec.append(poolcnt)
        rmcnt = l[40]
        vec.append(rmcnt)
        unitcnt = l[44]
        vec.append(unitcnt)
        yearbuild = l[47]
        vec.append(yearbuild)
        # the price key
        price = l[51]
        if len(price) > 0:
            price = float(price)
            for i in range(len(vec)):
                if len(vec[i]) > 0:
                    vec[i] = float(vec[i])
                else:
                    return []
            return [(price, vec)]
    return []

if __name__ == '__main__':
    # Get input/output files from user
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='File to load Amazon review data from')
    args = parser.parse_args()

    # Setup Spark
    conf = SparkConf().setAppName("tql")
    sc = SparkContext(conf=conf)

    tql_regression(sc, args.input)


# column id and meaning
# 0 parcelid
# 1 airconditioningtypeid
# 2 architecturalstyletypeid
# 3 basementsqft
# 4 bathroomcnt
# 5 bedroomcnt
# 6 buildingclasstypeid
# 7 buildingqualitytypeid
# 8 calculatedbathnbr
# 9 decktypeid
# 10 finishedfloor1squarefeet
# 11 calculatedfinishedsquarefeet
# 12 finishedsquarefeet12
# 13 finishedsquarefeet13
# 14 finishedsquarefeet15
# 15 finishedsquarefeet50
# 16 finishedsquarefeet6
# 17 fips
# 18 fireplacecnt
# 19 fullbathcnt
# 20 garagecarcnt
# 21 garagetotalsqft
# 22 hashottuborspa
# 23 heatingorsystemtypeid
# 24 latitude
# 25 longitude
# 26 lotsizesquarefeet
# 27 poolcnt
# 28 poolsizesum
# 29 pooltypeid10
# 30 pooltypeid2
# 31 pooltypeid7
# 32 propertycountylandusecode
# 33 propertylandusetypeid
# 34 propertyzoningdesc
# 35 rawcensustractandblock
# 36 regionidcity
# 37 regionidcounty
# 38 regionidneighborhood
# 39 regionidzip
# 40 roomcnt
# 41 storytypeid
# 42 threequarterbathnbr
# 43 typeconstructiontypeid
# 44 unitcnt
# 45 yardbuildingsqft17
# 46 yardbuildingsqft26
# 47 yearbuilt
# 48 numberofstories
# 49 fireplaceflag
# 50 structuretaxvaluedollarcnt
# 51 taxvaluedollarcnt
# 52 assessmentyear
# 53 landtaxvaluedollarcnt
# 54 taxamount
# 55 taxdelinquencyflag
# 56 taxdelinquencyyear
# 57 censustractandblock