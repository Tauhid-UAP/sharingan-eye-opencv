import cv2
import numpy as np

# if True, performs operations on still image
# if False, performs operations on own video frames
testing = False
#if True, finds centre of blob (eyeball)
# by calculating moments
# and then fits sharingan image
# otherwise, uses SimpleBlobDetector
use_blob_centre = True
# if True, marks face and eyes
mark = False

def frame_process(img, window_name, sharingan_img):
    # detect face
    face_coords = detect_face(img, face_cascade)
    if face_coords is not None:
        if mark:
            mark_object(face_coords, img)
        # get the coordinates of the face
        x, y, w, h = face_coords

        # midpoint required
        # otherwise both eye will be detected
        # as poorly adjusted left and right eyes
        # i.e, right eye will be detected as both left and right eyes
        # both eyes will be detected as right (or left) eye
        # but the left (or right) coordinates will be poorly adjusted
        face_width_midpoint = int(x + (w / 2))

        # partition face into left and right parts
        # on around midpoint
        face_left = img[y: y + h, x: face_width_midpoint]
        face_right = img[y: y + h, face_width_midpoint: x + w]

        # detect eyes from their respective sides
        lefteye_coords = detect_eye(face_left, lefteye_cascade)
        righteye_coords = detect_eye(face_right, righteye_cascade)

        threshold = 10
        if lefteye_coords is not None:
            eye_process(lefteye_coords, face_left, threshold, sharingan_img)
        if righteye_coords is not None:
            eye_process(righteye_coords, face_right, threshold, sharingan_img)

        cv2.imshow(window_name, img)

# detect and return the closest face coordinates
# closest defined by largest height
def detect_face(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords_array = classifier.detectMultiScale(gray_frame, 1.3, 5)

    # if multiple faces are detected
    # choose the closest (largest) one
    if len(coords_array) > 1:
        biggest = (0, 0, 0, 0)

        for coords in coords_array:
            if coords[3] > biggest[3]:
                biggest = coords
        # construct an array containing
        # the coordinates of the closest face
        closest = np.array([biggest], np.int32)
    elif len(coords_array) == 1:
        closest = coords_array
    else:
        return None
    
    # get the closest face coordinates
    face_coords = closest[0]
    
    return face_coords

# detect eye and return its coordinates on img
def detect_eye(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = classifier.detectMultiScale(gray_frame, 1.3, 5)

    eye_coords = None
    if(len(eye)):
        eye_coords = eye[0]

    return eye_coords

# draw BGR_color coloured rectangle
# around the region of the object (whose coordinates are target_coords) in the frame
# i.e, to mark eye, pass coordinates of eye as target_coords and face as frame
def mark_object(target_coords, frame, BGR_color=None):
    if not BGR_color:
        BGR_color = (255, 255, 0)

    x, y, w, h = target_coords
    cv2.rectangle(frame, (x, y), (x + w, y + h), BGR_color, 2)

# return the part of image img
# specified by coordinates coords
def coords_to_frame(coords, img):
    x, y, w, h = coords
    return img[y: y + h, x: x + w]

# concept of following function achieved from https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
# given an eye image img
# omit the eyebrows
def cut_eyebrows(img):
    height, width = img.shape[: 2]
    # eyebrow is approximately in the top quarter
    eyebrow_h = int(height / 4)

    return img[eyebrow_h: height, : width]

# concept of following function achieved from https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
# finds key points of blob (eyeball)
def blob_process(img, detector, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)

    return keypoints

# concept of following function achieved from https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# finds coordinates of the centre of blob (eyeball)
def find_blob_centre(img, threshold):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    
    moments = cv2.moments(img)
    # print('moments: ', moments)
    try:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        print('cx: ', cx)
        print('cy: ', cy)
    except:
        print('Division by zero!')
        return

    return (cx, cy)

# after detecting eye coordinates, perform necessary operations
# detect and mark eyeball
# call if blob_centre is False
def eye_process(coords, face_frame, threshold, sharingan_img):
    # mark the eye with a rectangle
    eye_rectangle_color = (255, 0, 255)
    if mark:
        mark_object(coords, face_frame, eye_rectangle_color)

    # make a frame of the eye region
    eye_frame = coords_to_frame(coords, face_frame)
    # cut eyebrows before blob detection
    eye_frame = cut_eyebrows(eye_frame)
    # print('eye_frame: ', eye_frame)
    print('eye_shape: ', eye_frame.shape)

    if use_blob_centre:
        # get coordinates of the centre of blob (eyeball)
        cx, cy = find_blob_centre(eye_frame, threshold)
        
        # eye_height required to determine proximity
        # to scale circle_radius
        eye_height = eye_frame.shape[0]
        
        # radius of the circle to draw
        # drawing is optional but the radius is required
        # circle_radius approximately 1/5 th of eye_height
        # closer the eye, greater the eye_height, greater the circle_radius
        circle_radius = int(eye_height / 5)

        print('sharingan_shape: ', sharingan_img.shape)

        # approximation of eyeball diameter
        # with respect to circle_radius
        eyeball_diameter = (circle_radius * 2) - 3

        # since size of sharingan_img is ambiguous
        # sharingan height and width required
        # to scale sharingan_img to eyeball
        sharingan_height = sharingan_img.shape[0]
        sharingan_width = sharingan_img.shape[1]

        # resize sharingan image
        # with respect to eyeball diameter
        sharingan_img = cv2.resize(
            sharingan_img,
            (0, 0),
            fx=(eyeball_diameter / sharingan_width),
            fy=(eyeball_diameter / sharingan_height)
        )
        print('sharingan_shape: ', sharingan_img.shape)

        # adjust starting position coordinates
        # of eyeball region in eye_frame
        x_start = cx - circle_radius
        y_start = cy - circle_radius + 5
        
        # isolate eyeball region as new frame
        # eyeball_frame and sharingan_img should have same shape
        # both having eyeball_diameter amount of pixels in either (x or y) direction
        eyeball_frame = eye_frame[y_start: y_start + eyeball_diameter, x_start: x_start + eyeball_diameter]
        
        # copy all pixels of sharingan_img
        # to pixel locations of eyeball_frame
        # i.e, copy sharingan to eyeball
        eyeball_frame[:, :] = sharingan_img
        
        # cv2.circle(eye_frame, (cx - 1, cy + 3), circle_radius + 1, (0, 0, 255), 1)

        return

    # find key points constructing the blob (eyeball)
    keypoints = blob_process(eye_frame, detector, threshold)
    print('keypoints: ', keypoints)

    # mark the key points
    # make a circle around eyeball
    cv2.drawKeypoints(eye_frame, keypoints, eye_frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# initialize face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize eye cascades
# use lefteye and righteye cascades
# on left and right side, respectively, of face
# for better adjustment
lefteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
righteye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# initialize blob detector parameters
# for eyeball detection
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

sharingan_img = cv2.imread('sharingan-images/three-tomoe.png')

def test_main():
    img = cv2.imread('opencv-face.jpeg')
    frame_process(img, 'Image', sharingan_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function with no purpose
# defined to pass to track bar
def nothing():
    pass

def main():
    cap = cv2.VideoCapture(0)
    window_name = 'My stream'
    cv2.namedWindow(window_name)
    cv2.createTrackbar('threshold', window_name, 0, 255, nothing)

    while True:
        _, frame = cap.read()
        frame_process(frame, window_name, sharingan_img)
        if cv2.waitKey(1) == ord('0'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


test_main() if testing else main()