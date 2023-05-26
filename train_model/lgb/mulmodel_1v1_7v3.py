


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
#from sklearn.externals import joblib

f = open('D:/Downloads/blood_train_data/test_data/sample_pos_multi_class_0605.csv',encoding='utf-8')
data = pd.read_csv(f,encoding='utf-8')
target = 'index'
IDcol = 'Unnamed: 0'
JBname = 'Result'
V1 = 'V1'
predictors = [x for x in data.columns if x not in [target,IDcol,JBname,V1]]#特征列筛选
data[predictors] = data[predictors].apply(pd.to_numeric,errors='coerce')

#标签编码
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Result'] = LabelEncoder().fit_transform(data['Result'])

#测试集训练集划分
train, validation = train_test_split(data, test_size = 0.3,random_state=2)

train_x = train[predictors]
train_y = train['Result']
validation_x = validation[predictors]
validation_y = validation['Result']

gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'multiclass',
                         metric = 'multi_logloss',
                         num_class = 9 ,
                         learning_rate = 0.1,
				         num_leaves = 285,
				         num_trees = 510,
			          	 num_threads = 16,
				         min_data_in_leaf = 0,
				         min_sum_hessian_in_leaf = 100,
                         is_unbalance = True
)
gbm.fit(train_x, train_y,eval_metric='logloss')


print('Start cross_val_score...')
# 测试机预测

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
test_auc = metrics.roc_auc_score(validation_y, test_predprob, multi_class='ovo')
recall = metrics.recall_score(validation_y, pre_y, average='weighted')
specificity = []
for i in range(9):
    tn = sum(confusion_matrix[j][j] for j in range(9) if j != i)
    fp = sum(confusion_matrix[j][i] for j in range(9) if j != i)
    specificity.append(tn / (tn + fp))
f1score = metrics.f1_score(validation_y, pre_y, average='weighted')
posvalue = []
for i in range(9):
    tp = confusion_matrix[i][i]
    fp = sum(confusion_matrix[j][i] for j in range(9) if j != i)
    posvalue.append(tp / (tp + fp))
nevalue = []
for i in range(9):
    tn = sum(confusion_matrix[j][j] for j in range(9) if j != i)
    fn = sum(confusion_matrix[i][j] for j in range(9) if j != i)
    nevalue.append(tn / (tn + fn))

lgb.plot_importance(gbm, max_num_features=30)
plt.title("Featureimportances")
plt.savefig('D:/Downloads/JWY/model/lgb/seed2_50_mul/Featureimportances_mulmodel.png',dpi=1200)

booster = gbm.booster_
importance = booster.feature_importance(importance_type='split')
feature_name = booster.feature_name()
# for (feature_name,importance) in zip(feature_name,importance):
# print (feature_name,importance)
feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
feature_importance.to_csv('D:/Downloads/JWY/model/lgb/seed2_50_mul/feature_importance_mulmodel.csv',index=False)


# 模型存储
joblib.dump(gbm, 'D:/Downloads/JWY/model/lgb/seed2_50_mul/data_pipei_out4var_18_mul.pkl')
# 模型加载
rt = pd.DataFrame([test_auc,recall,specificity,f1score,posvalue,nevalue],index=['test_auc','recall','specificity','f1score','posvalue','nevalue'])
print('rt',rt)
pd.DataFrame.to_csv(rt,'D:/Downloads/JWY/model/lgb/seed2_50_mul/data_pipei_out4var_18seed2_50testrt_mul.csv')


pickle.dump(gbm, open("D:/Downloads/JWY/model/lgb/seed2_50_mul/data_pipei_out4var_18_80pre_ageseed2_50test.pickle.dat", "wb"))
