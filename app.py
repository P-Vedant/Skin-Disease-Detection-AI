from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'UserData'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return "No file part", 400

    file = request.files['photo']

    if file.filename == '':
        return "No selected file", 400

    # Get the file extension
    ext = os.path.splitext(secure_filename(file.filename))[1]
    filename = f"f1{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)

    return f"File uploaded successfully as {filename}", 200

if __name__ == '__main__':
    app.run(debug=True)
