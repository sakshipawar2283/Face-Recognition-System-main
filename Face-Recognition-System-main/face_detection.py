
import cv2
# import numpy as np
import face_recognition as face_rec


def resize(img, size):
    width = int(img.shape[1] + size)
    height = int(img.shape[0] + size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


# img declaration
gauri = face_rec.load_image_file("sample_images/gauri.jpg")
gauri = cv2.cvtColor(gauri, cv2.COLOR_BGR2RGB)
gauri = resize(gauri, 0.30)
gauri_test = face_rec.load_image_file("sample_images/elonmusk.jpg")
gauri_test = resize(gauri_test, 0.30)
gauri_test = cv2.cvtColor(gauri_test, cv2.COLOR_BGR2RGB)


# finding face location

faceLocation_gauri = face_rec.face_locations(gauri)[0]
encode_gauri = face_rec.face_encodings(gauri)[0]
cv2.rectangle(gauri, (faceLocation_gauri[3], faceLocation_gauri[0]), (faceLocation_gauri[1], faceLocation_gauri[2]), (255, 0, 255), 3)

faceLocation_gauri_test = face_rec.face_locations(gauri_test)[0]
encode_gauri_test = face_rec.face_encodings(gauri_test)[0]
cv2.rectangle(gauri_test, (faceLocation_gauri_test[3], faceLocation_gauri_test[0]), (faceLocation_gauri_test[1], faceLocation_gauri_test[2]), (255, 0, 255), 3)


results = face_rec.compare_faces([encode_gauri], encode_gauri_test)
print(results)
cv2.putText(gauri_test, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('main_img', gauri)
cv2.imshow('test_img', gauri_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
