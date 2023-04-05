import tkinter as tk
import math

# create a function to calculate the delay
def calculate_delay():
    # get the input values from the entry widgets
    video_length_original = float(video_length_original_entry.get())
    video_length_detected = float(video_length_detected_entry.get())
    frame_rate_original = float(frame_rate_original_entry.get())
    frame_rate_detected = float(frame_rate_detected_entry.get())

    # calculate the number of frames in each video
    num_frames_original = math.floor(frame_rate_original * video_length_original)
    num_frames_detected = math.floor(frame_rate_detected * video_length_detected)

    # calculate the delay between the two videos
    delay = abs(num_frames_detected - num_frames_original) / frame_rate_original
    delay_frame = math.floor(frame_rate_detected * delay)
    delay_between_frame = math.floor(delay) / delay_frame

    # display the delay in the label widget
    delay_label.config(text="The delay between the two videos is: {} seconds.\nThe delay between frame is: {} milliseconds.".format(round(delay, 2), round(delay_between_frame*1000, 5)))

# create a tkinter window
window = tk.Tk()
window.title("Video Delay Calculator")

# add padding around the window and its contents
window.config(padx=10, pady=10)

# create labels and entry widgets for each input value
video_length_original_label = tk.Label(window, text="Original Video Length (sec): ")
video_length_original_entry = tk.Entry(window)
video_length_detected_label = tk.Label(window, text="Detected Video Length (sec): ")
video_length_detected_entry = tk.Entry(window)
frame_rate_original_label = tk.Label(window, text="Original Frame Rate (fps): ")
frame_rate_original_entry = tk.Entry(window)
frame_rate_detected_label = tk.Label(window, text="Detected Frame Rate (fps): ")
frame_rate_detected_entry = tk.Entry(window)

# create a button to calculate the delay
calculate_button = tk.Button(window, text="Calculate Delay", command=calculate_delay)

# create a label to display the delay
delay_label = tk.Label(window, text="")

# grid the labels, entry widgets, button, and delay label in the window
video_length_original_label.grid(row=0, column=0)
video_length_original_entry.grid(row=0, column=1)
video_length_detected_label.grid(row=1, column=0)
video_length_detected_entry.grid(row=1, column=1)
frame_rate_original_label.grid(row=2, column=0)
frame_rate_original_entry.grid(row=2, column=1)
frame_rate_detected_label.grid(row=3, column=0)
frame_rate_detected_entry.grid(row=3, column=1)
calculate_button.grid(row=4, column=0, columnspan=2)
delay_label.grid(row=5, column=0, columnspan=2)

# run the tkinter event loop
window.mainloop()
