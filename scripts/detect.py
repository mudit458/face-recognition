import os
import cv2
from time import sleep


def detect():
    fname = '../recognizer/trainingData.yml'
    if not os.path.isfile(fname):
        print('first train the data')
        exit(0)

    names = {}
    labels = []
    students = []

    face_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')
    # cap = cv2.VideoCapture('../test_videos/test3101.mp4')
    cap = cv2.VideoCapture(0)
    # cap.set(3,640) # set Width
    # cap.set(4,480) # set Height

    print('Total students :', names)

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # LOCAL BINARY PATTERNS HISTOGRAMS Face Recognizer

    recognizer.read(fname)  # read the trained yml file

    iddentified = set()

    num = 0
    SAMPLES = 10
    for _ in range(SAMPLES):
        ret, img = cap.read()
        # img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.rotate(img, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        final = cv2.medianBlur(equ, 3)

        faces = face_cascade.detectMultiScale(final, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            print('label:', label)
            print('confidence:', confidence)

            cv2.putText(img, str(label) + " " + str(confidence) + '%', (x + 2, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (150, 255, 0), 2)

            if (confidence <= 90):
                iddentified.add(label)

            f = 1
            cv2.imshow('Face Recognizer', img)
            k = cv2.waitKey(30) & 0xff
            # if cv2.waitKey(33) == ord('a'):
            num += 1
            if num > 100:
                cap.release()
                sleep(4)
                print('we are done!')
                f = 0
                break
        sleep(1)
    return iddentified


result = detect()
print(result)
