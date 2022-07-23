from flask import Flask, render_template, request
from joblib import load
app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def predict():
    request_type = request.method
    if request_type == 'GET':
        return render_template('index.html')
    else:
        model = load('app/k_means_model.joblib')

        pclass = float(request.form['pclass'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embark = request.form['embark']
        title = request.form['title']
        gender =request.form['gender']
        embark,title,gender = preprocessing(embark,title,gender)
        embark = float(embark)
        title = float(title)
        gender = float(gender)
        scale = scalar.transform([[age,fare]])
        age_scaled = float(scale[0][0])
        fare_scaled = float(scale[0][1])
        ans = model.predict([[pclass,age_scaled,fare_scaled,embark,title,gender]])
        if ans[0] == 0:
            ans_str = "didnt survived!"
        else:
            ans_str = "survived!"
        return render_template('index.html',prediction_text="The passenger {}".format(ans_str))

# def preprocessing(embark,title,gender):
#     def Embark(x):
#         if x == 'southampton':
#             return 0
#         elif x == 'cherbourg':
#             return 1
#         elif x == 'queenstown':
#             return 2
#     def title_no(x):
#         if x == 'Mr':
#             return 0
#         elif x == 'Miss':
#             return 1
#         elif x == 'Mrs':
#             return 2
#         elif x == 'Master':
#             return 3
#         elif x == 'Dr':
#             return 4
#         else:
#             return 0
#     embark = float(Embark(embark.lower()))
#     title = float(title_no(title))
#     if gender.lower() == 'male':
#         gender = float(1)
#     else:
#         gender = float(0)
#     return embark,title,gender