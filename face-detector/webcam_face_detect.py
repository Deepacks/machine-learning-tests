from cv2 import COLOR_BGR2GRAY, CascadeClassifier, VideoCapture, cvtColor, imshow, rectangle, waitKey

trained_face_data = CascadeClassifier(
    "./data/haarcascade_frontalface_default.xml")

webcam = VideoCapture(0)

while True:
    is_read_successful, frame = webcam.read()
    frame_greyscale = cvtColor(frame, COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(frame_greyscale)

    print("> face coordinates:", face_coordinates)

    for (x, y, w, h) in face_coordinates:
        rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    imshow("> Image: ", frame)
    key = waitKey(1)

    if key == 81 | key == 113:
        break

webcam.release()
print("✅ - Completed")
