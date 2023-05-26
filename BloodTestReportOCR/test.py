
import json
from io import StringIO
import base64
import bson
import cv2
import flask
import numpy
from PIL import Image

from imageFilter_v import ImageFilter
import re
import pickle as pk


if __name__ == "__main__":

    lgbm_2class_np = pk.load(open("model/gbm_2class_np.pkl", "rb"))
    lgbm_9class_positive = pk.load(open("model/gbm_9class_positive.pkl", "rb"))

    diseases = ['再生障碍性贫血', 
            '噬血细胞综合症', 
            '多发性骨髓瘤', 
            '急性淋巴细胞白血病',
            '急性髓样白血病',
            '淋巴瘤',
            '血小板减少性紫癜',
            '过敏性紫癜',
            '骨髓增生异常综合征']

    img_read = open('报告单/test1-9/test9再生障碍性贫血_01.jpg', 'rb')

    # print('{0},{1}'.format(base64image,fileName))
    def get_pred_X(report_data):
        X = [[]]
        X_label = ["MCH","MCHC","MCV","MPV","BASO","BASO%","EOS","HB","PDW%","PLT","EOS%","WBC","NEU%","NEUT","PCT","RDW%","LYMPH","LYM%","MON","RBC","MON%","HCT","Sex","Age"]
        
        for label in X_label:
            for i in range(22):
                alias = report_data['bloodtest'][i]['alias']
                value = report_data['bloodtest'][i]['value']
                if alias == label:
                    if re.match(r'^-?\d+(?:\.\d+)?$', value):
                        X[0].append(float(value))
                    else:
                        X[0].append(0)
        X[0].append(report_data['profile']['gender']=='男' if 0 else 1)
        X[0].append(int(report_data['profile']['age']))
        
        return X

    img = cv2.imdecode(numpy.frombuffer(img_read.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    report_data = ImageFilter(image=img).ocr(22)

    print(report_data)

    if report_data is None:
        data = {
            "error": 1,
        }
        print(jsonify(data))
    else:
        # prob = rf_model.predict(get_pred_X(report_data=report_data))
        # print(type(prob))
        # report_data['bloodtest'].append({
        #     'name':'九种血液病初筛结果及预警',
        #     'value': '非高危' if prob == 0 else '高危'
        # })
        # report_data['result'] = '非高危' if prob == 0 else '高危'
        # json_response = json.dumps(report_data, ensure_ascii=False, indent=4)
        # response = Response(json_response,content_type="application/json;charset=utf-8" )
        # return response

        for i in range(22):
            value = report_data['bloodtest'][i]['value']
            try:
                value = float(value)
            except ValueError:
               value = 0.0
           
            if value >= report_data['bloodtest'][i]['min'] and value <= report_data['bloodtest'][i]['max']:
                report_data['bloodtest'][i]['warn'] = 0
            else :
                report_data['bloodtest'][i]['warn'] = 1

        X1_p = {
        "MCH":29.8,
        "MCHC":333.0,
        "MCV":89.4,
        "MPV":9.00,
        "BAS":0.02,
        "BAS_P":0.20,
        "EOS":0.00,
        "HB":124.0,
        "PDW":9.50,
        "PLT":217,
        "EOS_P":0.00,
        "WBC":12.15,
        "NEUT_P":84.70,
        "NEUT":10.29,
        "PCT":0.190,
        "RDW":13.0,
        "LY":1.30,
        "LY_P":10.70,
        "MONO":0.54,
        "RBC":4.16,
        "MONO_P":4.40,
        "HCT":37.200,
        "Sex":1,
        "Age":57,
    }
        
        # X = [[29.8, 333.0, 89.4, 9.0, 0.02, 0.2, 0.0, 124.0, 9.5, 217, 0.0, 12.15, 84.7, 10.29, 0.19, 13.0, 1.3, 10.7, 0.54, 4.16, 4.4, 37.2, 1, 57]]
        # X = [[29,335,86.6,11.7,0.03,0.4,0.22,171,15.9,168,2.9,7.46,49,3.63,0.2,13.2,3.05,40.9,0.51,5.9,6.8,51.1,1,27]]

        X = get_pred_X(report_data)

        print(get_pred_X(report_data))
        print(f'data is '+str(X))
        
        predictions = lgbm_2class_np.predict(get_pred_X(report_data))
        # predictions = lgbm_2class_np.predict(X)

        print(predictions)

        if predictions[0] == 1:
            report_data['result'] = 0
            report_data['label'] = '健康'
        else:
            predictions = lgbm_9class_positive.predict(get_pred_X(report_data))
            # predictions = lgbm_9class_positive.predict(X)
            print(predictions)
            report_data['result'] = 1
            report_data['label'] = diseases[int(predictions[0])]

        