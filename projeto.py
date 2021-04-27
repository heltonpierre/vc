import cv2
<<<<<<< HEAD

webcam = cv2.VideoCapture(1)

#webcam = cv2.VideoCapture('1.mp4')
'''
ip = 'http://192.168.255.112:8080/video'
webcam = cv2.VideoCapture(ip)
'''
=======
webcam = cv2.VideoCapture(1)
>>>>>>> e1413e31e1117f25b996ba33880ceb8e859639fe
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

qtd_pessoas = int(0)

fgbg = cv2.createBackgroundSubtractorGMG()

while(True):
    s, video = webcam.read()

    video = cv2.flip(video, 180)

    faces = face_cascade.detectMultiScale(video, minNeighbors=20, minSize=(30, 30), maxSize=(400, 400))

    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 4)
        contador = str(faces.shape[0])
        cv2.putText(video, "Quantidade de Faces: " + contador, (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(contador)
        pessoas_dectadas = int(contador)
        print (f"Número de pessoas detectadas {pessoas_dectadas}")
        qtd_pessoas += pessoas_dectadas
        pessoas_dectadas = int(0)
        print (f"Número total de pessoas {qtd_pessoas}")
    
    cv2.imshow("Face Detection", video)

    if (cv2.waitKey(1) and 0xFF == ord('q')):
        break

webcam.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
