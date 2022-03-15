# View

from flask import Flask , jsonify , request
from Model_View import getPrediction

app = Flask(__name__)
@app.route('/predict_digit' , methods = ['POST'])

def predict_data() : 
    if not request.files : 
        return jsonify({
            'status' :  'Please submit a valid file',
            #  When task is successfully done it return 200 and when not it returns 400 
            'error_code' : 400
        })
    # fetching the data from the user input
    image = request.files.get('digit')
    predicted_value = getPrediction(image)

    return jsonify(predicted_value)

if __name__ == '__main__'  : 
    app.run(debug = True , port = 8080)
    