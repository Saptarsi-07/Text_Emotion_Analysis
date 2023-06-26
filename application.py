from flask import Flask,request, render_template,jsonify
from src.prediction_pipeline import CustomData,PredictPipeline


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
        data=CustomData(
            text=(request.form.get('text')),
            
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data['text'])

        result=pred

        return render_template('form.html',final_result=result)
    



if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)