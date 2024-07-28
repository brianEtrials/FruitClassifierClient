from flask import Flask, render_template, request, jsonify
import base64

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snapshot', methods=['POST'])
def snapshot():
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1]) #image data is what gets sent to model
    file_path = f"./shots/fruit.jpg"
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return jsonify(status="success", file_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)

