import cv2
import numpy as np
import dlib

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def find_avg_eyelid_distance(shape):
    eyelid_points = [[38, 42], [39, 41], [44, 48], [45, 47]]
    sum = 0
    for p1, p2 in eyelid_points:
        sum += find_distance_between_points(shape, p1, p2)
    return sum/4

def find_distance_between_points(shape, p1, p2):
    return ((shape[p1 - 1][0] - shape[p2 - 1][0])**2 + (shape[p1 - 1][0] - shape[p2 - 1][0])**2)**(1/2)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

THRESHOLD = 0.8
eye_mask = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]


cam = cv2.VideoCapture(0)

while (1):
    retval, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        for point in eye_mask:
            x, y = shape[point]
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        avg_eye_distance = find_avg_eyelid_distance(shape)
        print(avg_eye_distance)
        if (avg_eye_distance < THRESHOLD):
            print("CLOSED")


    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
