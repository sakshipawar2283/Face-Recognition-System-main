import os
import cv2
import numpy as np
import face_recognition as face_rec
from datetime import datetime
import pyttsx3 as textspeech
engine = textspeech.init()


def resize(img, size):
    width = int(img.shape[1] + size)
    height = int(img.shape[0] + size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


path = 'student_images'
studentimg = []
studentName = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # student_images/Ambolesir.jpg
    studentimg.append(curImg)
    studentName.append(os.path.splitext(cl)[0])


# print(studentName)

def findencoding(images):
    encoding_list = []
    for img in images:
        img = resize(img, 0.30)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        encoding_list.append(encodeimg)
    return encoding_list


encodeListKnown = findencoding(studentimg)
 # print(len(encodeListKnown))
 # print("ALL ENCODINGS COMPLETE!!!")


def markattendance(name):
    with open('attendance.csv', 'r+') as f:
        mydatalist = f.readline()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            time_now = datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tstr},{dstr}')



            statement = str('Welcome to class' + name)
            engine.say(statement)
            engine.runAndWait()



EncodeList = findencoding(studentimg)


vid = cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    facesInFrame = face_rec.face_locations(frames)
    encodeFacesInFrame = face_rec.face_encodings(frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_rec.compare_faces(encodeListKnown, encodeFace)
        facedis = face_rec.face_distance(encodeListKnown, encodeFace)
        # print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 2, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            import pandas as pd
            reading = pd.read_csv('attendance.csv')
            names = list(reading.loc[:, 'Name'])

            if name in names:
                pass
            else:
                print(name)
                markattendance(name)

    cv2.imshow('video', frame)
    if cv2.waitKey(10) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
