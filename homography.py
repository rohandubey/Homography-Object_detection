import cv2
import numpy as np

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)  # queryiamge
cap = cv2.VideoCapture(0)

#Features detectAndCompute
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)

# fEATURE MATHCING
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage

    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.73 * n.distance:
            good_points.append(m)
    # img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

    # Homography
    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        dst = np.array(dst, np.int32)
        convexhull = cv2.convexHull(dst)

        # transparent overlays
        frame_copy=frame.copy()
        homography = cv2.fillConvexPoly(frame, dst,(0, 0, 255))
        alpha = 0.5
        cv2.addWeighted(homography, alpha, frame_copy, 1 - alpha,	0, frame_copy)


        cv2.imshow("Homography", frame_copy)
    else:
        cv2.imshow("Homography", frame)

    # cv2.imshow("img3", img3)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
