from flask import Flask
'''
 It creates an instance of the Flask class, 
 which will be your WSGI (Web Server Gateway Interface) application.
'''
###WSGI Application
app1=Flask(__name__)

@app1.route("/")
def welcome():
    return "Welcome to this Flask course."

@app1.route("/index")
def index():
    return "Welcome to the index page"


@app1.route("/About")
def About():
    return "Welcome to the About page"

@app1.route("/Contact")
def Contact():
    return "Welcome to the Contact page"

if __name__=="__main__":
    app1.run(debug=True)