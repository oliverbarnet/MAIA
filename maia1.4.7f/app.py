from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import jetson_inference
import jetson_utils
import os
from collections import deque

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'some_random_secret_key'
  # Required for sessions

# Password for dashboard access
DASHBOARD_PASSWORD = " "

# For storing last 5 classification results (in memory only)
history = deque(maxlen=5)

# Load models ONCE
net_1 = jetson_inference.imageNet(argv=[
    '--model=/home/nvidia4/maia/models/tumor_detection/resnet18.onnx',
    '--input_blob=input_0',
    '--output_blob=output_0',
    '--labels=/home/nvidia4/maia/models/tumor_detection/labels.txt'
])
net_2 = jetson_inference.imageNet(argv=[
    '--model=/home/nvidia4/maia/models/tumor_classification/resnet18.onnx',
    '--input_blob=input_0',
    '--output_blob=output_0',
    '--labels=/home/nvidia4/maia/models/tumor_classification/labels.txt'
])
net_3 = jetson_inference.imageNet(argv=[
    '--model=/home/nvidia4/maia/models/alzheimers/resnet18.onnx',
    '--input_blob=input_0',
    '--output_blob=output_0',
    '--labels=/home/nvidia4/maia/models/alzheimers/labels.txt'
])
net_4 = jetson_inference.imageNet(argv=[
    '--model=/home/nvidia4/maia/models/covid/resnet18.onnx',
    '--input_blob=input_0',
    '--output_blob=output_0',
    '--labels=/home/nvidia4/maia/models/covid/labels.txt'
])
net_5 = jetson_inference.imageNet(argv=[
    "--model=/home/nvidia4/maia/models/fractures/resnet18.onnx",
    "--input_blob=input_0",
    "--output_blob=output_0",
    "--labels=/home/nvidia4/maia/models/fractures/labels.txt"
])
net_6 = jetson_inference.imageNet(argv=[
    "--model=/home/nvidia4/maia/models/EDC/resnet18.onnx",
    "--input_blob=input_0",
    "--output_blob=output_0",
    "--labels=/home/nvidia4/maia/models/EDC/labels.txt"
])

nets = {
    '2': ("Brain Tumor Detection", net_1),
    '3': ("Brain Tumor Classification", net_2),
    '4': ("Alzheimer's Classification", net_3),
    '5': ("COVID-19 Detection", net_4),
    '6': ("Fracture Detection", net_5),
    '7': ("Eye Disease Classification", net_6),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == DASHBOARD_PASSWORD:
            session['logged_in'] = True
            # Do NOT set session.permanent = True here
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error=True)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in', False):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['image']
        mode = request.form['mode']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(filename)

        img = jetson_utils.loadImage(filename)

        if mode not in nets:
            return jsonify({'error': 'Invalid mode'}), 400

        model_name, net = nets[mode]
        class_idx, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_idx)

        result = {
            'model': model_name,
            'prediction': class_desc,
            'confidence': f"{confidence*100:.2f}%",
            'id': class_idx,
            'filename': file.filename
        }

        history.appendleft(result)  # Add newest first
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify(list(history))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 403
    history.clear()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
