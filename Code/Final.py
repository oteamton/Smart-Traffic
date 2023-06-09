import cv2
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import numpy as np

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

with open("D:\project\yolo\classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

root = tk.Tk()

def open_file():
    file_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_path)
    process_video(cap)

def process_video(cap):
    start_time = time.time()
    net = cv2.dnn.readNet("D:\project\yolo\yolov4.weights", "D:\project\yolo\yolov4.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    obj_id = {}
    previous_centers = {}
    current_centers = {}
    fps_time = 0
    distance_r = 400
    # Initialize counters for each object class
    motorcycle_count = 0

    while True:
        start = time.time()
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        #farest green
        line_position1 = int(frame.shape[0] / 7) 
        cv2.line(frame, (0, line_position1), (frame.shape[1], line_position1), (0, 255, 0), 1)
        #middle yellow
        line_position2 = int(frame.shape[0] / 2)
        cv2.line(frame, (0, line_position2), (frame.shape[1], line_position2), (0, 165, 255), 1)
        #nearest red
        line_position3 = int(frame.shape[0] / 1.2)
        cv2.line(frame, (0, line_position3), (frame.shape[1], line_position3), (0, 0, 255), 1)
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
            color_a = (255,0,0)
            color = COLORS[int(classid) % len(COLORS)]
            # label = "%s : %f" % (class_names[classid], score)
            # not showing conf
            label = "%s" % (class_names[classid])
            
            x, y, w, h = box
            center = (int(x + w / 2), int(y + h / 2)) 
            current_centers[classid] = center
            f_color = (255,255,255)
            # arrow 
            arrow_len = int(max(w, h) * 1.2)
            arrow_start = (int(x + w / 2), y - arrow_len)
            arrow_end = (center[0], center[1] - arrow_len)
            f_color = (255,255,255)
            if classid in previous_centers:
                distance = cv2.norm(current_centers[classid], previous_centers[classid], cv2.NORM_L2)
                time_elapsed = 1 / (1 / (time.time() - start))
                distance_per_pixel = 0.1  # in meters
                speed_in_pixels_per_frame = distance / time_elapsed
                speed_in_meters_per_second = speed_in_pixels_per_frame * distance_per_pixel
                speed_in_km_per_hour = speed_in_meters_per_second * 3600 / 900
                label = label + " Speed: %.2f km/h" % speed_in_km_per_hour
                # check if the object is within the RoI and if its speed exceeds 50 km/h
                if line_position1 <= previous_centers[classid][1] <= line_position2 and \
                speed_in_km_per_hour > 45:
                    # easygui.msgbox("Speed Alert in zone 70 : %s km/h" % speed_in_km_per_hour)
                    cv2.putText(frame, "ALERT! Object speed > 45 km/h", (0, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    color = (0,165,255)
                    color_a = (0,165,255)
                    f_color = (1,1,1)
                if line_position2 <= previous_centers[classid][1] <= line_position3 and \
                speed_in_km_per_hour > 30:
                    # easygui.msgbox("Speed Alert in zone 50 : %s km/h" % speed_in_km_per_hour)
                    cv2.putText(frame, "ALERT! Object speed > 30 km/h", (0, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    color = (0,0,255)
                    color_a = (0,0,255)
                    f_color = (0,0,0)
            
            cv2.arrowedLine(frame, arrow_start, arrow_end, (color_a), 4)
            previous_centers[classid] = center
            cv2.rectangle(frame, box, color, 1)
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
            cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (f_color), 2)

        # # apply the mask to the frame
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
    if elapsed_time is not None:
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    cap.release()
    cv2.destroyAllWindows()

# root = tk.Tk()
# root.title("Video Processing")

# # Create a label for displaying the elapsed time
# elapsed_time_label = tk.Label(root, text="")
# elapsed_time_label.pack()

style = ttk.Style(root)
style.configure('W.TButton', font=('calibri', 20), foreground='#000', background='#4caf50', borderwidth='0', padx=20, pady=10)
open_button = ttk.Button(root, text="Open file", command=open_file, style='W.TButton')
open_button.pack()

root.mainloop()
