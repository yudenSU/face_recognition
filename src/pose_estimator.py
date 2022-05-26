# make funct that returns the pose estimation angle of the face
import numpy as np
import cv2
import face_recognition

def pose_estimator(image):
    image_height, image_width, image_colour = image.shape
    face_coor_2d = []
    face_coor_3d = []

    face_landmarks_list = face_recognition.face_landmarks(image)

    for key in face_landmarks_list:
        (x_coor_landmark, y_coor_landmar) = face_landmarks_list[key]

        x_coordinate, y_coordinate = int(x_coor_landmark * image_width), int(y_coor_landmar * image_height)

                    # Get the 2D Coordinates
        face_coor_2d.append([x_coordinate, y_coordinate])

                    # Get the 3D Coordinates
        face_coor_3d.append([x_coordinate, y_coordinate, landmark.z])       
            
        face_2d = np.array(face_2d, dtype=np.float64)



   	    # The camera matrix
        focal_length = 1 * image_width

        cam_matrix = np.array([ [focal_length, 0, image_height / 2],
                                    [0, focal_length, image_width / 2],
                                    [0, 0, 1]]) 


        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_coor_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360

        print(x,y)