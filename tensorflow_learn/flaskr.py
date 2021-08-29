# coding=utf-8
from flask import Flask,request,jsonify,redirect,url_for,render_template

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def mainPage():
    if request.method == 'POST':
        text = request.form['input']
        type_ = request.form['choose']
        if text:
            #params = {"text":text,"type_":type_}
            return redirect(url_for('print',text=text,type_=type_))
    return render_template('layout.html')

@app.route('/print',methods=['GET'])
def print():
    args = request.args
    text = args.get("text")
    type_name = args.get("type_")
    return jsonify({"status":0,"result":text[0],"type_name":type_name})

if __name__=='__main__':
    app.run('127.0.0.1',5000,debug=True)
