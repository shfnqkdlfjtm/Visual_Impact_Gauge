import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = r"C:\Users\samsung\Downloads\chessboard.avi"
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare a 3D box for simple AR (modify to create a cube)
cube_lower = board_cellsize * np.array([[3.5, 2.5, 0], [5.5, 2.5, 0], [5.5, 4.5, 0], [3.5, 4.5, 0],
                                        [3.5, 2.5, -1], [5.5, 2.5, -1], [5.5, 4.5, -1], [3.5, 4.5, -1]])
cube_upper = board_cellsize * np.array([[3.5, 2.5, -1], [5.5, 2.5, -1], [5.5, 4.5, -1], [3.5, 4.5, -1],
                                        [3.5, 2.5, -2], [5.5, 2.5, -2], [5.5, 4.5, -2], [3.5, 4.5, -2]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the cube on the image
        cube_lower_projected, _ = cv.projectPoints(cube_lower, rvec, tvec, K, dist_coeff)
        cube_upper_projected, _ = cv.projectPoints(cube_upper, rvec, tvec, K, dist_coeff)
        for i in range(4):
            cv.line(img, tuple(map(int, cube_lower_projected[i].ravel())),
                    tuple(map(int, cube_lower_projected[(i + 1) % 4].ravel())), (255, 0, 0), 2)
            cv.line(img, tuple(map(int, cube_upper_projected[i].ravel())),
                    tuple(map(int, cube_upper_projected[(i + 1) % 4].ravel())), (0, 0, 255), 2)
            cv.line(img, tuple(map(int, cube_lower_projected[i].ravel())),
                    tuple(map(int, cube_upper_projected[i].ravel())), (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()