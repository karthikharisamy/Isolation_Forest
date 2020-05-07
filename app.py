from flask import Flask,render_template,url_for,request
from flask_material import Material

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/load')
def index():
    return render_template("indexs.html")
     


@app.route('/predict',methods=["POST"])
def analyze():
	if request.method == 'POST':
		Annual_Income= request.form['Annual_Income']
		Spending_Score = request.form['Spending_Score']
		Model_choice = request.form['Model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [Annual_Income,Spending_Score]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

		# Reloading the Model
		if Model_choice == 'logitmodel':
		    logit_model = joblib.load('savesss.pckl')
		    result_prediction = logit_model.predict(ex1)
		    anomaly_score=logit_model.decision_function(ex1)
		    if result_prediction[0]==1:
		        label ='The datapoint is not a outlier or an anomaly'
		    elif result_prediction[0]==-1:
		        label ='The datapoint is an outlier or anomaly'
		        
		elif Model_choice == 'knnmodel':
			knn_model = joblib.load('IrisModel_knn.pckl')
			result_prediction = knn_model.predict(ex1)
		elif Model_choice == 'svmmodel':
			knn_model = joblib.load('IrisModel_svm.pckl')
			result_prediction = knn_model.predict(ex1)

	return render_template('indexs.html', Annual_Income=Annual_Income,
		Spending_Score=Spending_Score,
		clean_data=clean_data,
		result_prediction=label,
		model_selected=Model_choice,anomaly_score=anomaly_score)


if __name__ == '__main__':
	app.run(debug=True)