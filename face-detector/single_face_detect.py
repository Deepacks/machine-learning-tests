from cv2 import COLOR_BGR2GRAY, CascadeClassifier, cvtColor, imread, imshow, rectangle, waitKey

trained_face_data = CascadeClassifier(
    "./data/haarcascade_frontalface_default.xml")

test_image_1 = imread("./test/portrait.jpeg")
test_image_1_greyscale = cvtColor(test_image_1, COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(test_image_1_greyscale)

print("> face coordinates:", face_coordinates[0])

for (x, y, w, h) in face_coordinates:
    rectangle(test_image_1, (x, y), (x + w, y + h), (0, 0, 255), 2)

imshow("> Image: ", test_image_1)
waitKey()
print("✅ - Completed")
