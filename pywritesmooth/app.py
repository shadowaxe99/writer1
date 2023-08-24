from flask import Flask, render_template, request
import lstm_model as handwriting_smoothing

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['user_input']
        result = handwriting_smoothing.smooth(user_input)
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)