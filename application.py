from flask import Flask,request, render_template,jsonify

import pickle
import pandas as pd


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    

    else:
        data = request.form.get('text')
        new_data=pd.Series(data=data)

        

        with open('classifier.pkl','rb') as file_obj:
          model=pickle.load(file_obj)
        
        result=model.predict_proba(new_data)

        final_result={
            'Negative: ':result[:][0][0],
            'Positive: ':result[:][0][1]
        }





        return render_template('form.html',final_result=final_result)
    



if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)