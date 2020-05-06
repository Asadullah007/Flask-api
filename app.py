from flask import Flask, jsonify, render_template, request,url_for
import pyrebase
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
import json
from datetime import datetime

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/covid19'
db = SQLAlchemy(app)


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    consent_for_participating = db.Column(db.String(80), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    country = db.Column(db.String(20), nullable=True)
    patient_id = db.Column(db.String(20), nullable=False)
    Corona_Test = db.Column(db.String(20), nullable=False)
    age = db.Column(db.String(20), nullable=False)
    #timestamp = db.Column(db.String(20), nullable=False)
    prs_fcs = db.Column(db.String(50), nullable=True)
    prs_sob = db.Column(db.String(50), nullable=True)
    prs_nwc = db.Column(db.String(50), nullable=True)
    prs_st = db.Column(db.String(50), nullable=True)
    prs_ba = db.Column(db.String(50), nullable=True)
    prs_los = db.Column(db.String(50), nullable=True)
    prs_none = db.Column(db.String(50), nullable=True)
    mh_acld = db.Column(db.String(50), nullable=True)
    mh_p = db.Column(db.String(50), nullable=True)
    mh_dwc = db.Column(db.String(50), nullable=True)
    mh_dc = db.Column(db.String(50), nullable=True)
    mh_chf = db.Column(db.String(50), nullable=True)
    mh_eo = db.Column(db.String(50), nullable=True)
    mh_none = db.Column(db.String(50), nullable=True)
    smoker = db.Column(db.String(50), nullable=False)
    patient_location = db.Column(db.String(50), nullable=False)
     
    




@app.route('/', methods = ['GET', 'POST'])
def appform():
    if(request.method=='POST'):
        '''Add entry to the database'''
        consent_for_participating = request.form.get('consent_for_participating')
        gender = request.form.get('gender')
        country = request.form.get('country')
        patient_id = request.form.get('patient_id')
        Corona_Test = request.form.get('Corona_Test')
        age = request.form.get('age')
        prs_fcs = request.form.get('prs_fcs')
        prs_sob = request.form.get('prs_sob')
        prs_nwc = request.form.get('prs_nwc')
        prs_st  = request.form.get('prs_st ')
        prs_ba = request.form.get('prs_ba')
        prs_los = request.form.get('prs_los')
        prs_none = request.form.get('prs_none')

        mh_acld = request.form.get('mh_acld') 
        mh_p = request.form.get('mh_p')
        mh_dwc = request.form.get('mh_dwc')
        mh_dc = request.form.get('mh_dc')
        mh_chf = request.form.get('mh_chf')
        mh_eo = request.form.get('mh_eo')
        mh_none = request.form.get('mh_none')
        smoker = request.form.get('smoker')
        patient_location = request.form.get('patient_location')
         
        entry = Contact(consent_for_participating=consent_for_participating,
                         gender=gender,
                         country=country,
                         patient_id=patient_id,
                         Corona_Test=Corona_Test,
                         age=age,
                         prs_fcs=prs_fcs,
                         prs_sob=prs_sob,
                         prs_nwc=prs_nwc,
                         prs_st=prs_st,
                         prs_ba=prs_ba,
                         prs_los=prs_los,
                         prs_none=prs_none,
                         mh_acld=mh_acld,
                         mh_p=mh_p,
                         mh_dwc=mh_dwc,
                         mh_dc=mh_dc,
                         mh_chf=mh_chf,
                         mh_eo=mh_eo,
                         mh_none=mh_none,
                         smoker=smoker,
                         patient_location=patient_location
                         )
        db.session.add(entry)
        db.session.commit()
        return render_template('index.html')
        
    return render_template('app.html')



@app.route('/', methods=["POST","GET"])
def signup():

    if request.method == "POST":
        userData = [{
    "name": request.form['consent_for_participating'],
    "email": request.form['gender'],
    }]

        
        return jsonify(userData)


@app.route('/signup' , methods=["POST","GET"])
def index():
    
    if request.method == "POST":
        userData = [{
    "patient_reported_symptoms": request.form['prs_fcs'],
    "medical_history": request.form['prs_fcs'],
    "age": request.form['age'],
    "gender": request.form['gender'],
    "smoker": request.form['smoker']
    
    }]

        
        return jsonify(userData)

    return  render_template('app.html')

@app.route('/html')
def html():
     
     return render_template('index.html')

#@app.route('/signup', methods=["POST","GET"])
#def signup():
#    if request.method == "POST":
 #       name1 = request.form.get('name')
  #      email1 = request.form['email']
   #     return jsonify( '''The User name is: {}
    #               The User email is: {},. '''.format(name1,email1))
'''        
######## Post man Wala code ################
 '''   
@app.route('/json_example', methods = ["POST"])
def json_example():
    req_data = request.get_json()
    name = req_data['name']
    email = req_data['email']
    return 'hello {} , your email is {}'.format(name,email)



app.run(debug=True)