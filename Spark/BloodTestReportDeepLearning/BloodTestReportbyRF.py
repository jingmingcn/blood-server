# -*- coding: cp936 -*-
#����Spark�������Ѫ������鱨�����ѧϰϵͳ
#2016.12.14

from __future__ import print_function

import sys
import math
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":

    sc = SparkContext(appName="BloodTestReportPythonRandomForestExample")

    # ��ȡ����.
    print('Begin Load Data File!')
    sexData = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata_sex.txt")
    ageData = MLUtils.loadLabeledPoints(sc, "LabeledPointsdata_age.txt")
    #print(data.collect())
    print('Data File has been Loaded!')
    #for(d in data.take(3)):
    #   print(d)
    accuracySex = []
    accuracyAge = []
    for i in range(0,100):
        # ����������ָ�Ϊ9:1���ֱ���Ϊѵ�����ݣ�training����Ԥ�����ݣ�test��.
        sexTraining, sexTest = sexData.randomSplit([0.9, 0.1])
        ageTraining, ageTest = ageData.randomSplit([0.9, 0.1])

        # ѵ�������ģ��.
        #print('Begin RandomForest tranning!')
        sexModel = RandomForest.trainClassifier(sexTraining,numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)
        ageModel = RandomForest.trainClassifier(ageTraining,numClasses=1000, categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)
        #print('Trainning over!')
        # ��test���ݽ���Ԥ�⣬���Ԥ��׼ȷ��.
        sexPredictionAndLabel = sexTest.map(lambda p: p.label).zip(sexModel.predict(sexTest.map(lambda x: x.features)))
        agePredictionAndLabel = ageTest.map(lambda p: p.label).zip(ageModel.predict(ageTest.map(lambda x: x.features)))
        #print(predictionAndLabel.collect())
        accuracySex.append(1.0 * sexPredictionAndLabel.filter(lambda (x, v): x == v).count() / sexTest.count())
        accuracyAge.append(1.0 * agePredictionAndLabel.filter(lambda (x, v): abs((x-v)<=5)).count() / ageTest.count())
    #AVG��ƽ����  MSE��������
    SexRDD = sc.parallelize(accuracySex)
    AgeRDD = sc.parallelize(accuracyAge)
    SexPAAVG = SexRDD.reduce(lambda x,y:x+y)/SexRDD.count()
    AgePAAVG = AgeRDD.reduce(lambda x,y:x+y)/AgeRDD.count()
    SexPAMSE = math.sqrt(SexRDD.map(lambda x:(x - SexPAAVG)*(x - SexPAAVG)).reduce(lambda x,y:x+y)/SexRDD.count())
    AgePAMSE = math.sqrt(AgeRDD.map(lambda x:(x - AgePAAVG)*(x - AgePAAVG)).reduce(lambda x,y:x+y)/AgeRDD.count())
    #print(sum(accuracySex) / len(accuracySex))
    #print(sum(accuracyAge) / len(accuracyAge))

    print('Sex Prediction Accuracy AVG:{}'.format(SexPAAVG))
    print('Sex Prediction Accuracy MSE:{}'.format(SexPAMSE))
    print('AGE Prediction Accuracy AVG:{}'.format(AgePAAVG))
    print('AGE Prediction Accuracy MSE:{}'.format(AgePAMSE))

    output = open('RandomForestResult.txt', 'w')
    output.write('Sex Prediction Accuracy AVG is:' + str(SexPAAVG) + "\n")
    output.write('Sex Prediction Accuracy MSE is:' + str(SexPAMSE) + "\n")
    for i in accuracySex:
        output.write(str(i)+",")
    output.write("\n")
    output.write('Age Prediction Accuracy AVG is:' + str(AgePAAVG) + "\n")
    output.write('Age Prediction Accuracy AVG is:' + str(AgePAMSE) + "\n")
    for i in accuracyAge:
        output.write(str(i) + ",")
    output.write("\n")
    output.close()
    
