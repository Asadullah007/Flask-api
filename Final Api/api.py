from flask import Flask, jsonify, request, make_response
from flask_restful import Resource, Api
import json
from datetime import datetime

app = Flask(__name__)
api = Api(app)

# /api/virufy -> all data will be posted here and we will return the response

class App(Resource):
    def post(self):
# "age", "gender", "smoker", "patient_reported_symptoms", "medical_history", cough audio, breath audio, finger video
        print(request.args)
        age = request.args.get('age')
        gender = request.args.get('gender')
        smoker = request.args.get('smoker')
        symptoms = request.args.get('reported_symptoms')
        medical_history = request.args.get('medical_history')

        response = {"age": age, "gender": gender,
         "smoker": smoker, "reported_symptoms": symptoms,
         "medical_history": medical_history}

        return make_response(jsonify(response), 200)

api.add_resource(App, "/api/virufy")
app.run(port=3000,debug=True)
