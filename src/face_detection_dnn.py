import numpy.core.multiarray
import cv2
import sys
from embed import embedder
#from pose_estimator import pose_estimator
import face_recognition
import numpy as np

# Detects a face via a specified camera input and returns a 128d embedding of the face.
def face_detect_dnn_register(camera_input):

    DNN_model_path = "src/deploy.prototxt"
    caffe_model_path = "src/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    net = cv2.dnn.readNetFromCaffe(DNN_model_path, caffe_model_path)
    confidence_threshold = 0.7

    # set selected camera as video input
    source = cv2.VideoCapture(camera_input, cv2.CAP_DSHOW)

    # window to display input
    cam_window = 'camera input'
    cv2.namedWindow(cam_window, cv2.WINDOW_NORMAL)

    # 3d model for pose estimation
    model_3d_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

    # distance coefficients assuming no lens distortio
    dist_coefficients_matrix = np.zeros((4,1))

    has_frame, frame = source.read()

    focal_length = frame.shape[1]
    center = (frame.shape[1]/2, frame.shape[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )



    # uses "esc" as exit key
    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        # if frame does not exist, exit window
        if not has_frame:
            break
        # mirror the image (flip over x axis)
        frame = cv2.flip(frame,1)
        height = frame.shape[0]
        width = frame.shape[1]

        # creates a blob, the parameters below are specified by the requirements of the DNN model used
        camera_input_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [103, 117, 123], swapRB = False, crop = False)
        
        # Run a model
        net.setInput(camera_input_blob)
        detected_faces = net.forward()

        for i in range(detected_faces.shape[2]):
            confidence = detected_faces[0, 0, i, 2]
            if confidence > confidence_threshold:
                start_x = int(detected_faces[0, 0, i, 3] * width)
                start_y = int(detected_faces[0, 0, i, 4] * height)
                end_x = int(detected_faces[0, 0, i, 5] * width)
                end_y = int(detected_faces[0, 0, i, 6] * height)

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0))
                # crops out the range of interest/face
                roi_color = frame[start_y:end_y,start_x:end_x]
                # ensures the roi exist
                if roi_color.size > 0:
#                    vec = embedder(roi_color)
#                    pose_estimator(roi_color)
                    #print(x_pose_coor, y_pose_coor)
                    #print(vec)

                    image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    face_landmarks_list = face_recognition.face_landmarks(image)

                    # for face_landmarks in face_landmarks_list:
                    #     for facial_feature in face_landmarks.keys():
                    #         for value in face_landmarks[facial_feature]:
                    #             roi_color = cv2.circle(roi_color, value, radius=0, color=(0, 0, 255), thickness=-1)

                    if face_landmarks_list != []:
                        left_eye_corner = face_landmarks_list[0]["left_eye"][0]
                        right_eye_corner = face_landmarks_list[0]["right_eye"][3]

                        face_landmarks_points = [
                                        face_landmarks_list[0]["nose_bridge"][3],
                                        face_landmarks_list[0]["chin"][8],
                                        left_eye_corner,
                                        right_eye_corner,
                                        face_landmarks_list[0]["bottom_lip"][6],
                                        face_landmarks_list[0]["bottom_lip"][0]
                                    ]
                        face_landmarks_points = np.array(face_landmarks_points, dtype="double")

                        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_3d_points, face_landmarks_points, camera_matrix, dist_coefficients_matrix)
                        rmat, jac = cv2.Rodrigues(rotation_vector)
                            
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)


                        left_eye_x = left_eye_corner[0]
                        left_eye_y = left_eye_corner[1]
                        right_eye_x = right_eye_corner[0]
                        right_eye_y = right_eye_corner[1]
                
                        delta_x = right_eye_x - left_eye_x
                        delta_y = right_eye_y - left_eye_y
                        
                        # Slope of line formula
                        angle = np.arctan(delta_y / delta_x) 
                        
                        # Converting radians to degrees
                        angle = (angle * 180) / np.pi 
                        print(angle)
                        if angles[1] <=  -10:
                            yaw = "looking right"
                        elif angles[1] >= 10:
                            yaw = "looking left"
                        else:
                            yaw = "forward"
                        if angles[0] < -175:
                            pitch = "looking down"
                        elif angles[0] > 170:
                            pitch = "looking up"
                        else:
                            pitch = "forward"

                        cv2.putText(frame, yaw, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
                        cv2.putText(frame, pitch, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)


                    #x = np.arctan2(Qx[2][1], Qx[2][2])
                    #y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2]))) * 360
                    #z = np.arctan2(Qz[0][0], Qz[1][0])
                    #print("ThetaX: ", x)
                    #print("ThetaY: ", y)
                    #print("ThetaZ: ", z)
                                #print(x)
                                #print(y)


                    #print(face_landmarks_list)
                    
                    cv2.imwrite("roi.png", roi_color)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # TODO: checks for glasses

                # TODO: checks that the user is looking at the camera

        label = "Face detection"
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow(cam_window, frame)

    source.release()
    cv2.destroyWindow(cam_window)

if __name__ == '__main__':
    webcam_code = 0
    face_detect_dnn_register(webcam_code)