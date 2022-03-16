from cv2 import COLOR_BGR2GRAY, CascadeClassifier, cvtColor, imread, imshow, rectangle, waitKey

trained_face_data = CascadeClassifier(
    "./data/haarcascade_frontalface_default.xml")

test_image_2 = imread("./test/portrait_two_people.jpeg")
test_image_2_greyscale = cvtColor(test_image_2, COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(test_image_2_greyscale)

print("> face coordinates:")
for coordinates in face_coordinates:
    print(" -", coordinates)

for (x, y, w, h) in face_coordinates:
    rectangle(test_image_2, (x, y), (x + w, y + h), (0, 0, 255), 2)

imshow("> Image: ", test_image_2)
waitKey()
print("✅ - Completed")
