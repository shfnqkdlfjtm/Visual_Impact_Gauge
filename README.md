# Visual_Impact_Gauge
Camera Pose Estimation and AR

## 프로그램 및 기능 설명
이 프로그램은 카메라의 위치와 방향을 추정하여 체스보드 상의 3D 객체를 시각화하는 작업을 수행한다. 프로그램은 다음과 같은 기능을 포함함:

1. 주어진 비디오 파일에서 프레임을 읽어온다.
2. 체스보드 패턴을 찾기 위해 이미지에서 코너를 검출한다.
3. 찾은 코너를 사용하여 카메라의 위치와 방향을 추정한다.
4. 추정된 카메라 위치와 방향을 기반으로 체스보드 상에 3D 객체를 시각화한다.
5. 카메라의 위치 정보를 이미지에 출력한다.
6. ESC 키를 누를 때까지 이러한 프로세스를 반복한다.
7. 이를 통해 프로그램은 카메라가 체스보드에 대해 어떻게 위치하고 있는지를 실시간으로 추적하고 시각화하여 사용자에게 제공한다.

## 코드
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

## 결과
![AR](https://github.com/shfnqkdlfjtm/Chessboard_Hex_Float/assets/144716487/14115f07-2e48-44e2-937d-278ffd05772e)
