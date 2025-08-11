from flask import Flask, render_template, request, jsonify
import jetson_inference
import jetson_utils
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        file = request.files['image']
        mode = request.form['mode']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = jetson_utils.loadImage(filename)

        if mode == '2':
            net = net_1
        elif mode == '3':
            net = net_2
        elif mode == '4':
            net = net_3
        elif mode == '5':
            net = net_4
        elif mode == "6":
            net = net_5
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        class_idx, confidence = net.Classify(img)
        class_desc = net.GetClassDesc(class_idx)

        return jsonify({
            'prediction': class_desc,
            'confidence': f"{confidence*100:.2f}%",
            'id': class_idx
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
