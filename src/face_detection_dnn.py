import cv2
import sys
from embed import embedder

def face_detect_dnn():

    # default input is set to acccess the user's webcam
    cam_input = 0

    # check if any other camera is selected to be used
    if len(sys.argv) > 1:
        cam_input = sys.argv[1]

    DNN_model_path = "face_recognition/src/deploy.prototxt"
    caffe_model_path = "face_recognition/src/res10_300x300_ssd_iter_140000_fp16.caffemodel"

    net = cv2.dnn.readNetFromCaffe(DNN_model_path, caffe_model_path)
    confidence_threshold = 0.7

    # set selected camera as video input
    source = cv2.VideoCapture(cam_input, cv2.CAP_DSHOW)

    # window to display input
    cam_window = 'camera input'
    cv2.namedWindow(cam_window, cv2.WINDOW_NORMAL)


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
                roi_color = frame[start_x:start_y,end_x:end_y]
                # ensures the roi exist
                if roi_color.size > 0:
                    vec = embedder(roi_color)
                    print(vec)
                    cv2.imwrite("roi_color.png", roi_color)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # TODO: checks for glasses

                # TODO: checks that the user is looking at the camera

        label = "Face detection"
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow(cam_window, frame)

    source.release()
    cv2.destroyWindow(cam_window)

if __name__ == '__main__':
    face_detect_dnn()