from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np 


import pickle

app = Flask(__name__)

random = pickle.load(open('co2_random.pkl','rb'))
decision = pickle.load(open('co2_decision.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset)
        return render_template("preview.html",df_view = df)


@app.route('/prediction')
def prediction():
    return render_template("prediction.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        make = request.form['make']
        Vehicle_Class = request.form['Vehicle_Class']
        Engine_Size = request.form['Engine_Size']
        Cylinders = request.form['Cylinders']
        Transmission = request.form['Transmission'] 
        Fuel_Type = request.form['Fuel_Type']
        Fuel_Consumption_City = request.form['Fuel_Consumption_City']
        Fuel_Consumption_Hwy = request.form['Fuel_Consumption_Hwy']
        Fuel_Consumption_comb = request.form['Fuel_Consumption_comb']
        Fuel_Consumption_comb_mpg = request.form['Fuel_Consumption_comb_mpg']
        
        
        model = request.form['Model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [make,Vehicle_Class,Engine_Size,Cylinders,Transmission,Fuel_Type,Fuel_Consumption_City,Fuel_Consumption_Hwy,Fuel_Consumption_comb,Fuel_Consumption_comb_mpg]
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if model == 'RandomForestClassifier':
           result_prediction = random.predict(ex1)
           
            
        elif model == 'DecisionTreeClassifier':
          result_prediction = decision.predict(ex1)
           
           
        
        if result_prediction > 4.5:
            result = 'BEST'
        else:
            result = 'WORST'    

          

    return render_template('prediction.html', prediction_text= result,result_prediction = int(result_prediction), model = model)

@app.route('/performance')
def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
	app.run(debug=True)
