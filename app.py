from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import the model processing function from modeltest.py
from modeltest import process_image
from modelsalienttest import process_salient_image
from fpdf import FPDF

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
                repair_cost = process_image(filepath, app.config['RESULTS_FOLDER'])

                processed_salient_od_filepath = process_salient_image(filepath, app.config['RESULTS_FOLDER'])
                results.append({
                    'original': 'uploads/' + filename,  # Use relative path for web access
                    'processed': 'results/vis/' + filename,  # Use relative path for web access static\results\vis\0000test.jpg
                    'boxesonly': 'results/cost/'+ filename,
                    'salient_od':'results/mask/' + filename,
                    'repair_cost': repair_cost
                })
                #results.append({'original': filepath, 'processed': processed_file_path})
                print("The result is:",results)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        return render_template('results.html', results=results)
    return render_template('object_detection.html')

@app.route('/salient_detection', methods = ['GET', 'POST'])
def salient_detection():
    if request.method == 'POST':
        files = request.files.getlist('images')
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                processed_filepath = process_salient_image(filepath, app.config['RESULTS_FOLDER'])
                print('processed file saved to:', processed_filepath)
                print(processed_filepath)
                results.append({
                    'original': 'uploads/' + filename,
                    'processed': 'results/' + 'mask_'+ filename
                })
                print(results)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        return render_template('salient_results.html', results = results)
    return render_template('salient_detection.html')


@app.route('/live_video')
def live_video():
    return render_template('live_video.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/generate_report')
def generate_report():
    # Logic to generate a report based on the results
    return jsonify({'message': 'Report generated successfully'})

@app.route('/view_statistics')
def view_statistics():
    # Logic to calculate and display statistics
    return jsonify({'message': 'Statistics displayed'})

@app.route('/call_help')
def call_help():
    # Logic to initiate a help call
    return jsonify({'message': 'Help has been called'})

@app.route('/get_recommendations')
def get_recommendations():
    # Logic to provide recommendations
    return jsonify({'message': 'Recommendations provided'})



@app.route('/download_report')
def download_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Damage Report", ln=True, align='C')
    # Add more data to your PDF here
    response = make_response(pdf.output(dest='S').encode('latin1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
