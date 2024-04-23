# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import os

# # Assuming process_image function will be defined, using a placeholder here
# def process_image(image_path):
#     # This is a placeholder function. Implement your actual image processing here.
#     # For example, it could return the path to a processed image or processing results.
#     return image_path  # Placeholder return

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit per file

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/object_detection', methods=['GET', 'POST'])
# def object_detection():
#     if request.method == 'POST':
#         files = request.files.getlist('images')
#         if not files:
#             return jsonify({'error': 'No files provided'}), 400
#         results = []
#         for file in files:
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)
#                 # Process the image through your ML model here
#                 result = process_image(filepath)
#                 results.append({'original': filename, 'processed': result})
#             else:
#                 return jsonify({'error': 'Invalid file type'}), 400
#         return render_template('results.html', results=results)
#     return render_template('object_detection.html')

# @app.route('/salient_detection')
# def salient_detection():
#     return render_template('salient_detection.html')

# @app.route('/live_video')
# def live_video():
#     return render_template('live_video.html')

# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import the model processing function from modeltest.py
from modeltest import process_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit per file

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/object_detection', methods=['GET', 'POST'])
def object_detection():
    if request.method == 'POST':
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print("Saved original file to:", filepath) 
                # Process the image through your ML model here
                processed_file_path = process_image(filepath, app.config['RESULTS_FOLDER'])
                print("Processed file saved to:", processed_file_path)
                results.append({
                    'original': 'uploads/' + filename,  # Use relative path for web access
                    'processed': 'results/vis/' + filename  # Use relative path for web access static\results\vis\0000test.jpg
                })
                #results.append({'original': filepath, 'processed': processed_file_path})
                print("The result is:",results)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        return render_template('results.html', results=results)
    return render_template('object_detection.html')

@app.route('/salient_detection')
def salient_detection():
    return render_template('salient_detection.html')

@app.route('/live_video')
def live_video():
    return render_template('live_video.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
