import pickle
# import pandas as pd

if __name__ == '__main__':

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

    X2_n = {
        "MCH":29.1,
        "MCHC":323,
        "MCV":90.1,
        "MPV":10.5,
        "BAS":0.01,
        "BAS_P":0.2,
        "EOS":0.05,
        "HB":135,
        "PDW":12.7,
        "PLT":229,
        "EOS_P":0.9,
        "WBC":5.53,
        "NEUT_P":48.6,
        "NEUT":2.69,
        "PCT":0.24,
        "RDW":12.1,
        "LY":2.56,
        "LY_P":46.3,
        "MONO":0.22,
        "RBC":4.64,
        "MONO_P":4,
        "HCT":41.8,
        "Sex":1,
        "Age":49
    }

    X = [[29.8,333.0,89.4,9.00,0.02,0.20,0.00,124.0,9.50,217,0.00,12.15,84.70,10.29,0.190,13.0,1.30,10.70,0.54,4.16,4.40,37.200,0,57]]
    model = pickle.load(open('model/gbm_2class_np.pkl','rb'))
    print(model)
    predictions = model.predict(X)
    print(predictions)