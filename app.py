from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load các model từ folder models-warehouse
def load_models():
    dtc = pickle.load(open('./models-warehouse/model_DTC_Drug.sav', 'rb'))
    logistic = pickle.load(open('./models-warehouse/model_Logistic_Drug.sav', 'rb'))
    knn = pickle.load(open('./models-warehouse/model_KNN_Drug.sav', 'rb'))
    bayes = pickle.load(open('./models-warehouse/model_NaiveBayes_Drug.sav', 'rb'))
    return dtc, logistic, knn, bayes

dtc_model, logistic_model, knn_model, bayes_model = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Lấy dữ liệu từ Form
    age = int(request.form['age'])
    blood_scaled = int(request.form['blood_pressure'])
    cholesterol_scaled = int(request.form['cholesterol'])
    na_to_k = float(request.form['na_to_k'])
    sex = request.form['sex']

    # 2. Xử lý One-Hot Encoding cho Sex
    sex_female = 1 if sex == 'Female' else 0
    sex_male = 1 if sex == 'Male' else 0

    # 3. Tạo DataFrame khớp với cấu hình training
    columns = ['Age', 'Blood_Pressure', 'Cholesterol', 'Sodium_to_Potassium', 'Sex_Female', 'Sex_Male']
    feature_sample = pd.DataFrame([[age, blood_scaled, cholesterol_scaled, na_to_k, sex_female, sex_male]], 
                                 columns=columns)

    # 4. Dự đoán
    res_dtc = dtc_model.predict(feature_sample)[0]
    res_logistic = logistic_model.predict(feature_sample)[0]
    res_knn = knn_model.predict(feature_sample)[0]
    res_bayes = bayes_model.predict(feature_sample)[0]

    predictions = {
        'dtc': res_dtc,
        'logistic': res_logistic,
        'knn': res_knn,
        'bayes': res_bayes
    }

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)