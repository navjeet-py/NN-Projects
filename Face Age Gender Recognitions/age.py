import requests
import json
import cv2

# Detect all human faces present in a given image and try to guess their age, gender and emotion state via their facial shapes

# Target image: Feel free to change to whatever image holding as many human faces as you want
img = 'https://i.ibb.co/Sxm6VG3/mp.jpg'
key = '2232ca4b9015b852c265e91cd59b7919'
cvImg = cv2.imread('../mp.jpg', 1)
cv2.waitKey(0)
req = requests.get('http://api.pixlab.io/facemotion', params={
    'img': img,
    'key': key,
})
reply = req.json()
if reply['status'] != 200:
    print(reply['error'])
    exit()

total = len(reply['faces'])  # Total detected faces
print(str(total) + " faces were detected")
# Extract each face now
for face in reply['faces']:
    cord = face['rectangle']
    print('Face coordinate: width: ' + str(cord['width']) + ' height: ' + str(cord['height']) + ' x: ' + str(
        cord['left']) + ' y: ' + str(cord['top']))
    left = cord['left']
    top = cord['top']
    right = left + cord['width']
    bottom = top + cord['height']
    age = str(face['age'])
    gender = str(face['gender']).capitalize()
    # Guess emotion
    emot = ''
    for emotion in face['emotion']:
        if emotion['score'] > 0.5:
            print("Emotion - " + emotion['state'] + ': ' + str(emotion['score']))
            emot = emot + emotion['state']
    emot.capitalize()

    print("Age ~: " + age)
    print("Gender: " + gender)
    cv2.rectangle(cvImg, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

    # Draw a label with a name below the face
    cv2.rectangle(cvImg, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(cvImg, gender + ' ' + age + ' ' + emot, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)
cv2.imshow('Image', cvImg)
cv2.waitKey(0)
