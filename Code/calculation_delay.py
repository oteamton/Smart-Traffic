import math

# take user input for video length and frame rate
video_length_original = float(input("Enter the original video length in seconds: "))
video_length_detected = float(input("Enter the detected video length in seconds: "))
frame_rate_original = float(input("Enter the original video frame rate: "))
frame_rate_detected = float(input("Enter the detected video frame rate: "))

# calculate number of frames in each video
num_frames_original = math.floor(frame_rate_original * video_length_original)
num_frames_detected = math.floor(frame_rate_detected * video_length_detected)

# calculate delay between the two videos
delay = abs(num_frames_detected - num_frames_original) / frame_rate_original
delay_frame = math.floor(frame_rate_detected * delay)
delay_between_frame = math.floor(delay) / delay_frame
# print the delay
print("The delay between the two videos is:", round(delay, 2), "seconds.")
print("The delay between frame is:", round(delay_between_frame*1000, 5), "milliseconds.")
