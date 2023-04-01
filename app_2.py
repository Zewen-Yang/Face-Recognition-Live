from flask import Flask, render_template, Response, send_from_directory, url_for
import face_recognition
import cv2
import numpy as np
import os
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from flask_uploads import UploadSet, IMAGES, configure_uploads
from wtforms import SubmitField

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

app=Flask(__name__)

app.config["SECRET_KEY"] = "secret"
app.config["UPLOADED_PHOTOS_DEST"] = "photos"
photos = UploadSet("photos")
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
        FileAllowed(photos, "Only images are allowed"),
        FileRequired("File field should not be empty")
        ]
    )
    submit = SubmitField("Upload")


app=Flask(__name__)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)

PHOTO_PATH = "photos"
# Load a sample picture and learn how to recognize it.
Zewen_image = face_recognition.load_image_file(os.path.join(PHOTO_PATH, "Zewen/Zewen.jpg"))
Zewen_face_encoding = face_recognition.face_encodings(Zewen_image)[0]

# Load a second sample picture and learn how to recognize it.
Voula_image = face_recognition.load_image_file(os.path.join(PHOTO_PATH, "Voula/Voula.jpg"))
Voula_face_encoding = face_recognition.face_encodings(Voula_image)[0]
# Load a second sample picture and learn how to recognize it.
Antonio_image = face_recognition.load_image_file(os.path.join(PHOTO_PATH, "Antonio/Antonio.jpg"))
Antonio_face_encoding = face_recognition.face_encodings(Antonio_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    Zewen_face_encoding,
    Voula_face_encoding,
    Antonio_face_encoding,
]
known_face_names = [
    "Zewen",
    "Voula",
    "Antonio"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def GenerateFrames():
    while True:
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Only process every other frame of video to save time
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/video_feed')
def video_feed():
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/photos/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)
    else:
        file_url = None
    return render_template("upload.html", form=form, file_url=file_url)


if __name__=='__main__':
    app.run(debug=True)