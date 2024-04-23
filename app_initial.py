from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        save_path = 'uploads'
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(save_path, filename)
        file.save(file_path)
        # Here you would typically process the file through your model
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
