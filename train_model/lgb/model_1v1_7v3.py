
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import gc
from sklearn import metrics
#from sklearn.externals import joblib
import joblib
import pickle
import matplotlib
from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from matplotlib import pyplot
import graphviz
from sklearn.preprocessing import LabelEncoder

f = open('D:/Downloads/blood_train_data/test_data/data_1V1.csv',encoding='utf-8')
data = pd.read_csv(f,encoding='utf-8')
target = 'target'
IDcol = 'Unnamed: 0'
JBname = 'Result'
predictors = [x for x in data.columns if x not in [target,IDcol,JBname]]
data[predictors] = data[predictors].apply(pd.to_numeric,errors='coerce')
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

train, validation = train_test_split(data, test_size = 0.3,random_state=2)
train_x = train[predictors]
train_y = train['target']
validation_x = validation[predictors]
validation_y = validation['target']

del train,validation
gc.collect()

train = lgb.Dataset(train_x, train_y)
test = lgb.Dataset(validation_x, validation_y, reference=train)

gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'binary_logloss',
                         learning_rate = 0.1,
				         num_leaves = 255,
				         num_trees = 500,
				         num_threads = 16,
				         min_data_in_leaf = 0,
				         min_sum_hessian_in_leaf = 100,
                         is_unbalance = True
)
gbm.fit(train_x, train_y,eval_metric='logloss')

print('Start cross_val_score...')
# 测试机预测

#scores = cross_val_score(gbm,train_x, train_y, cv=10)
#scores2 = scores.mean()
print('Start predicting...')

test_predprob = gbm.predict_proba(validation_x)
pre_y= gbm.predict(validation_x)
classify_report = metrics.classification_report(validation_y, pre_y)
confusion_matrix = metrics.confusion_matrix(validation_y, pre_y)
overall_accuracy = metrics.accuracy_score(validation_y, pre_y)
acc_for_each_class = metrics.precision_score(validation_y, pre_y, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(validation_y, pre_y)
print('classify_report', classify_report)
print('confusion_matrix', confusion_matrix)
print('acc_for_each_class', acc_for_each_class)
#print('average_accuracy: {0:f}'.format(average_accuracy))
#print('overall_accuracy: {0:f}'.format(overall_accuracy))
#print('score: {0:f}'.format(score))
test_auc = metrics.roc_auc_score(validation_y,test_predprob[:,1])#验证集上的auc值
recall = metrics.recall_score(validation_y, pre_y)
specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
f1score = metrics.f1_score(validation_y, pre_y, average='weighted')
posvalue = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
nevalue = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
fpr1,tpr1,thresholds1 = metrics.roc_curve(validation_y, test_predprob [:,1],pos_label = 1)
roc_auc_new = metrics.auc(fpr1,tpr1)
plt.plot(fpr1, tpr1, lw=2,label='ROC curve of class {0} (area = {1:0.2f})'''.format('aml', roc_auc_new))
plt.savefig('D:/Downloads/JWY/model/lgb/seed2_50/testroc.png',dpi=1200)
plt.figure(figsize=(12,6))
lgb.plot_importance(gbm, max_num_features=30)
plt.title("Featureimportances")
plt.savefig('D:/Downloads/JWY/model/lgb/seed2_50/Featureimportances.png',dpi=1200)

booster = gbm.booster_
importance = booster.feature_importance(importance_type='split')
feature_name = booster.feature_name()
# for (feature_name,importance) in zip(feature_name,importance):
# print (feature_name,importance)
feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
feature_importance.to_csv('D:/Downloads/JWY/model/lgb/seed2_50/feature_importance.csv',index=False)


# 模型存储
joblib.dump(gbm, 'D:/Downloads/JWY/model/lgb/seed2_50/data_pipei_out4var_18.pkl')
# 模型加载
rt = pd.DataFrame([test_auc,recall,specificity,f1score,posvalue,nevalue],index=['test_auc','recall','specificity','f1score','posvalue','nevalue'])
print('rt',rt)
pd.DataFrame.to_csv(rt,'D:/Downloads/JWY/model/lgb/seed2_50/data_pipei_out4var_18seed2_50testrt.csv')

pre = pd.DataFrame([test_predprob[:,1],validation_x['Sex'],validation_x['Age'],validation_y],index = ['prob','gender','age','test_y'])
pre1 = pd.DataFrame(pre.values.T, index=pre.columns, columns=pre.index)
pd.DataFrame.to_csv(pre1,'D:/Downloads/JWY/model/lgb/seed2_50/data_pipei_out4var_18_80seed2_50testpre_age.csv')

pickle.dump(gbm, open("D:/Downloads/JWY/model/lgb/seed2_50/data_pipei_out4var_18_80pre_ageseed2_50test.pickle.dat", "wb"))
