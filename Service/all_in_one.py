import cv2
import dlib
import numpy as np
import os
from keras.models import model_from_json, Model
import tempfile
import base64


def get_layer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
    raise Exception("Layer with name " + name + " does not exist")


def load_model(model_json_path, model_h5_path, layer_names):
    with open(model_json_path, "r") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(model_h5_path)
        layer_output = []
        for lname in layer_names:
            layer_output += [get_layer(model, lname).output]

        output = Model(inputs=model.inputs, output=layer_output)
        return output


def predict_image(image_64, image_type='RGB'):
    model = load_model("models/allinone.json", "models/allinone.h5", ["age_estimation", "smile", "gender_probablity"])

    binary_image = base64.b64decode(image_64)

    f = tempfile.NamedTemporaryFile()
    f.write(binary_image)

    if image_type == 'RGB':
        image = cv2.imread(f.name)
        IMAGE_TYPE = 'RGB'
    else:
        image = cv2.imread(f.name)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        IMAGE_TYPE = 'L'

    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    all_in_one_reslut = []
    for i in range(len(faces)):
        face = faces[i]
        face_image = image[face.top():face.bottom(), face.left():face.right()]
        face_image = cv2.resize(face_image, (227, 227))
        face_image = face_image.astype(np.float32) / 255
        predictions = model.predict(face_image.reshape(-1, 227, 227, 3))
        image = cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), thickness=3)
        age_estimation = predictions[0][0]
        smile_detection = predictions[1][0]
        gender_probablity = predictions[2][0]

        age = np.argmax(age_estimation)
        smile = np.argmax(smile_detection)
        gender = np.argmax(gender_probablity)

        if smile == 0:
            smile = "False"
        else:
            smile = "True"
        if gender == 0:
            gender = "Female"
        else:
            gender = "Male"
        all_in_one_reslut.append([face.top(), face.bottom(), face.left(), face.right(), age, smile, gender])
    return all_in_one_reslut

# with open("imgs/adele.png", 'rb') as f:
#     img = f.read()
#     image_64 = base64.b64encode(img).decode('utf-8')
#
# predict_image(image_64=image_64)
