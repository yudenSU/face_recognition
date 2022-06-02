# make funct that returns the pose estimation angle of the face
import numpy as np
import cv2
import face_recognition
from bucket import bucket

def pose_estimator(frame, roi_color, face_landmarks_list):
    image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

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

    focal_length = frame.shape[1]
    center = (frame.shape[1]/2, frame.shape[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    face_landmarks_list = face_recognition.face_landmarks(image)

    # this 3 for loop section is only here to draw key landmarks on the face, removing it would 
    # be a good idea for performance
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            for value in face_landmarks[facial_feature]:
                roi_color = cv2.circle(roi_color, value, radius=0, color=(0, 0, 255), thickness=-1)

    # the  variable posture needs to 3 marking that the yaw, pitch and row of the face is 
    #facing the camera before the image of roi is taken
    posture = 0
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
        tilt = np.arctan(delta_y / delta_x) 
                        
        # Converting radians to degrees
        tilt = (tilt * 180) / np.pi 
        if tilt <=  -1.5:
            tilt_description = "tilted left"
        elif tilt >= 1.5:
            tilt_description = "tilted right"
        else:
            tilt_description = "upright"
            posture += 1

        if angles[1] <=  -10:
            yaw_description = "looking right"
        elif angles[1] >= 10:
            yaw_description = "looking left"
        else:
            yaw_description = "forward"
            posture += 1

        if angles[0] < -175:
            pitch_description = "looking down"
        elif angles[0] > 170:
            pitch_description = "looking up"
        else:
            pitch_description = "forward"
            posture += 1

            cv2.putText(frame, yaw_description, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
            cv2.putText(frame, pitch_description, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
            cv2.putText(frame, tilt_description, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
    if posture == 3:
        return True
    else:
        return False
                    