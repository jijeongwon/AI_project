import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]])

    print(arr)
    print(model)
    pred = model.predict(arr)
    print(pred)
    threshold = 20000  # 임계값 설정
    if pred <= threshold:
        return render_template('result_low.html', data=pred)
    elif threshold <= pred <= 26000:
        return render_template('result_mid.html', data=pred)
    elif pred >= 26000:
        return render_template('result_high.html', data=pred)


if __name__ == "__main__":
    model = pickle.load(open("C://projects/project/models/RFmodel.pkl", 'rb'))

    app.run(debug=True)
