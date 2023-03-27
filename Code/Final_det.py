import cv2
import time
import easygui
import numpy as np
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

with open("D:\Space\Python\yolo\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture("D:\\Space\\Python\\video_traffic\\vid5.mp4")
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
    for (classid, score, box) in zip(classes, scores, boxes):
        if class_names[classid] != "motorcycle":
            continue
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        
        x, y, w, h = box
        center = (int(x + w / 2), int(y + h / 2))
        current_centers[classid] = center
        
        if classid in previous_centers:
            distance = cv2.norm(current_centers[classid], previous_centers[classid], cv2.NORM_L2)
            time_elapsed = 1 / (1 / (time.time() - start))
            distance_per_pixel = 0.1  # in meters
            speed_in_pixels_per_frame = distance2 / time_elapsed
            speed_in_meters_per_second = speed_in_pixels_per_frame * distance_per_pixel
            speed_in_km_per_hour = speed_in_meters_per_second * 3600 / 120000
            label = label + " Speed: %.2f km/h" % speed_in_km_per_hour
            # check if the object is within the RoI and if its speed exceeds 50 km/h
            if line_position1 <= previous_centers[classid][1] <= line_position2 and \
            speed_in_km_per_hour > 70:
                # easygui.msgbox("Speed Alert in zone 70 : %s km/h" % speed_in_km_per_hour)
                cv2.putText(frame, "ALERT! Object speed > 70 km/h", (0, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if line_position2 <= previous_centers[classid][1] <= line_position3 and \
            speed_in_km_per_hour > 50:
                # easygui.msgbox("Speed Alert in zone 50 : %s km/h" % speed_in_km_per_hour)
                cv2.putText(frame, "ALERT! Object speed > 50 km/h", (0, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        previous_centers[classid] = center
        cv2.rectangle(frame, box, color, 2)
        # Get label size
        (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        if box[1] - h_label > 0:
            if box[0] + w_label < frame.shape[1]:
                label_pos = (box[0], box[1]-5)
            else:
                label_pos = (frame.shape[1]-w_label-5, box[1]-5)
        else:
            if box[0] + w_label < frame.shape[1]:
                label_pos = (box[0], box[1]+h_label+5)
            else:
                label_pos = (frame.shape[1]-w_label-5, box[1]+h_label+5)
    
        # Draw label background
        cv2.rectangle(frame, (label_pos[0]-3, label_pos[1]-h_label-3), (label_pos[0]+w_label+3, label_pos[1]+3), color, cv2.FILLED)
        
        # Draw label
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # apply the mask to the frame
    # roi_frame = cv2.bitwise_and(frame, mask)
    fps = "FPS: %.2f " % (1 / (time.time() - start))
    cv2.putText(frame, fps, (0, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)

    cv2.imshow("output", frame)
    # cv2.imshow("RoI", roi_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elapsed_time = time.time() - start_time
# print elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
cap.release()
cv2.destroyAllWindows()
