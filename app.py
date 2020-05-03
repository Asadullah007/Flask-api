from flask import Flask, jsonify, render_template, request , url_for 

app = Flask(__name__)


@app.route('/signup', methods=["POST","GET"])
def signup():

    if request.method == "POST":
        userData = [{
    "name": name,
    "email": email,
    }]

        for i in userData:
            return jsonify(i)


@app.route('/')
def index():
    return  render_template('app.html')

@app.route('/about')
def message():
    for i in userData:
        if(i['email'] == 'abd'):
            return jsonify(i)

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
        
@app.route('/forexample', methods=["POST","GET"])
def for_example():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form['email']
        return  '''The User name is: {}
                   The User email is: {},. '''.format(name,email)
   
    return '''<form method ='POST'>
    name <input typye = "text" name = "name">
    email <input typye = "text" name = "email">
    <input type = "submit">
    </form>'''
######## Post man Wala code ################
    
@app.route('/json_example', methods = ["POST"])
def json_example():
    req_data = request.get_json()
    name = req_data['name']
    email = req_data['email']
    return 'hello {} , your email is {}'.format(name,email)



app.run(debug=True)