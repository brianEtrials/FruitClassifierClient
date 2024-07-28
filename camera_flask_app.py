from flask import Flask, render_template, Response, request
import cv2
import os

global capture
capture = 0

# Create shots directory to save pictures
if not os.path.exists('./shots'):
    os.makedirs('./shots')

# Instantiate Flask app  
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

def frames():
    while True:
        success, frame = camera.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
           break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot', methods=['POST'])
def snapshot():
    global capture
    if request.form.get('click') == 'Capture':
        success, frame = camera.read()
        if success:
            file_path = f"./shots/fruit.jpg"
            cv2.imwrite(file_path, frame)
    return render_template('index.html', classification = 'apple')

if __name__ == '__main__':
    app.run(debug=False)

camera.release()
cv2.destroyAllWindows()
