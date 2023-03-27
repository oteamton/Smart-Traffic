import cv2
import time
import easygui
import numpy as np

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

with open("D:\Space\Python\yolo\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("D:\\Space\\Python\\video_traffic\\vid3.mp4")
start_time = time.time()
net = cv2.dnn.readNet("D:\Space\Python\yolo\yolov4.weights", "D:\Space\Python\yolo\yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

obj_id = {}
previous_centers = {}
current_centers = {}
fps_time = 0
distance2 = 500
# Initialize counters for each object class
motorcycle_count = 0
fps_limit = 30 # Maximum FPS to display

while True:
    start = time.time()
    (grabbed, frame) = cap.read()
    if not grabbed:
        break
    #farest green
    line_position1 = int(frame.shape[0] / 7) 
    cv2.line(frame, (0, line_position1), (frame.shape[1], line_position1), (0, 255, 0), 2)
    #middle yellow
    line_position2 = int(frame.shape[0] / 2)
    cv2.line(frame, (0, line_position2), (frame.shape[1], line_position2), (0, 255, 255), 2)
    #nearest red
    line_position3 = int(frame.shape[0] / 1.50)
    cv2.line(frame, (0, line_position3), (frame.shape[1], line_position3), (0, 0, 255), 2)
    # area = (int(frame.shape[0] / 0.5) - int(frame.shape[0] / 0.1)) * frame.shape[1]
    vertices = np.array([[(0, line_position1), (frame.shape[1], line_position1),
                      (frame.shape[1], line_position2), (0, line_position2),
                      ]], dtype=np.int32)
    vertices2 = np.array([[(0, line_position2), (frame.shape[1], line_position2),
                      (frame.shape[1], line_position3), (0, line_position3),
                      ]], dtype=np.int32)
    # # create a mask with the same size as the frame
    # mask = np.zeros_like(frame)
    # # fill the polygon with a color of 50% opacity
    # alpha = 0  # Set the desired alpha value
    # cv2.fillPoly(mask, [vertices], (0, 255, 0, int(alpha * 0)))
    # cv2.fillPoly(mask, [vertices2], (0, 0, 255, int(alpha * 0))) 
    
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    counter = 0
    for (classid, score, box) in zip(classes, scores, boxes):
        if class_names[classid] != "motorcycle":
                continue
        if isinstance(classid, (list, np.ndarray)): 
            classid = classid[0]
        label = "%s %d" % (class_names[classid], counter)
        counter += 1
        color = COLORS[classid % len(COLORS)]
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Check if object is a motorcycle and is within the detection area
        if class_names[classid] == "motorcycle" and box[1] > line_position2 and box[3] < line_position3:
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            current_centers[classid] = (x_center, y_center)

            # Check if object has crossed the line and update counter accordingly
            if classid not in obj_id:
                obj_id[classid] = y_center
            else:
                if y_center < obj_id[classid] - distance2:
                    obj_id[classid] = y_center
                    motorcycle_count += 1
    
    # Draw line to show detection area
    cv2.line(frame, (0, line_position2), (frame.shape[1], line_position2), (0, 255, 255), 2)
    cv2.line(frame, (0, line_position3), (frame.shape[1], line_position3), (0, 0, 255), 2)

    # Display motorcycle count
    cv2.putText(frame, "Motorcycle Count: " + str(motorcycle_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        
    # Add the FPS as text on top of the frame
    fps = "FPS: %.2f " % (1 / (time.time() - start))
    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow("Traffic Detection", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()