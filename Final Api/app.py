from flask import Flask, jsonify, request, make_response
# from .text import *
app = Flask(__name__)

# /api/virufy -> all data will be posted here and we will return the response

@app.route('/api/virufy', methods=['POST'])
def post():
    age = request.form.get('age')
    gender = request.form.get('gender')
    smoker = request.form.get('smoker')
    symptoms = request.form.getlist('reported_symptoms')
    medical_history = request.form.getlist('medical_history')

    response = {"age": age, "gender": gender,
     "smoker": smoker, "reported_symptoms": symptoms,
     "medical_history": medical_history}



    return make_response(jsonify(response), 200)

@app.route('/api/virufy', methods=['GET'])
def get():
    message = { 'message': 'This api just has endpoints for POST request' }
    return make_response(jsonify(message), 404)

app.run(port=5000, debug=True)
