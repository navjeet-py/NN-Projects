import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
import json

def add_face(img):
    name = img[:img.index('.')]
    face = fr.load_image_file(img)
    encoding = fr.face_encodings(face)[0].tolist()
    with open("faces.json", "r+") as file:
        data = json.load(file)
        data.update({name: encoding})
        file.seek(0)
        json.dump(data, file)

def get_encoded_faces():
    file = open('faces.json', 'r').read()
    encoded = json.loads(file)
    for i in encoded.keys():
        encoded[i] = np.array(encoded[i])
    return list(encoded.values()), list(encoded.keys())

faces_encoded,known_face_names = get_encoded_faces()

def classify_face(im):

    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    img = cv2.imread(im, 1)

    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    print(unknown_face_encodings[0].tolist())
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15),fontFace=font, fontScale=1.0, color=(255, 255, 255), thickness=2)
    print(face_names)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    return 0

def classify_folder(folderName):
    for i in os.listdir(folderName):
        if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg'):
            classify_face(folderName+'/'+i)


# add_face('test/Taylor Alison Swift.jpg')
classify_face('test/Taylor Alison Swift.jpg')