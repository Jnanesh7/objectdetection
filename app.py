from flask import Flask, render_template, Response, request
import cv2
import json

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load class names
with open("dnn_model/classes.txt", "r") as f:
    classes = [line.strip() for line in f]

# Global variable to store the selected object
selected_class = None

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html', classes=classes)

# Route to update the selected class
@app.route('/select_class', methods=['POST'])
def select_class():
    global selected_class
    selected_class = request.json.get('selected_class')
    return {"status": "success", "selected_class": selected_class}

# Video stream generator
def generate_frames():
    global selected_class
    cam = cv2.VideoCapture(0)
    CONF_THRESHOLD, NMS_THRESHOLD = 0.5, 0.4

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Perform detection
        class_ids, scores, bboxes = model.detect(frame, confThreshold=CONF_THRESHOLD, nmsThreshold=NMS_THRESHOLD)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            x, y, w, h = bbox
            class_name = classes[class_id]

            # Only show the bounding box for the selected class
            if class_name == selected_class:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
                cv2.putText(frame, f"{class_name} {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
