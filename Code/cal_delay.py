import math
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFormLayout

class VideoDelayCalculator(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # create input widgets
        self.video_length_original_label = QLabel("Original Video Length (sec): ")
        self.video_length_original_entry = QLineEdit()
        self.video_length_detected_label = QLabel("Detected Video Length (sec): ")
        self.video_length_detected_entry = QLineEdit()
        self.frame_rate_original_label = QLabel("Original Frame Rate (fps): ")
        self.frame_rate_original_entry = QLineEdit()
        self.frame_rate_detected_label = QLabel("Detected Frame Rate (fps): ")
        self.frame_rate_detected_entry = QLineEdit()

        # create calculate button and connect to calculate_delay method
        self.calculate_button = QPushButton("Calculate Delay")

        # create label to display delay
        self.delay_label = QLabel()

        # create form layout and add widgets to it
        form_layout = QFormLayout()
        form_layout.addRow(self.video_length_original_label, self.video_length_original_entry)
        form_layout.addRow(self.video_length_detected_label, self.video_length_detected_entry)
        form_layout.addRow(self.frame_rate_original_label, self.frame_rate_original_entry)
        form_layout.addRow(self.frame_rate_detected_label, self.frame_rate_detected_entry)
        form_layout.addWidget(self.calculate_button)
        form_layout.addWidget(self.delay_label)

        # set form layout and window properties
        self.setLayout(form_layout)
        self.setWindowTitle("Video Delay Calculator")
        self.setGeometry(100, 100, 400, 300)

        # connect calculate button to calculate_delay method
        self.calculate_button.clicked.connect(self.calculate_delay)

    def calculate_delay(self):
        # get input values
        video_length_original = float(self.video_length_original_entry.text())
        video_length_detected = float(self.video_length_detected_entry.text())
        frame_rate_original = float(self.frame_rate_original_entry.text())
        frame_rate_detected = float(self.frame_rate_detected_entry.text())

        # calculate delay
        num_frames_original = math.floor(frame_rate_original * video_length_original)
        num_frames_detected = math.floor(frame_rate_detected * video_length_detected)
        delay = abs(num_frames_detected - num_frames_original) / frame_rate_original
        delay_frame = math.floor(frame_rate_detected * delay)
        
        # check for division by zero
        if delay_frame == 0:
            self.delay_label.setText("Error: Cannot calculate delay between frames. Detected frame rate may be incorrect.")
            return
        
        delay_between_frame = math.floor(delay) / delay_frame    

        # set delay label text
        self.delay_label.setText("The delay between the two videos is: {} seconds.\nThe delay between frame is: {} milliseconds.".format(round(delay, 2), round(delay_between_frame*1000, 5)))

if __name__ == '__main__':
    app = QApplication([])
    window = VideoDelayCalculator()
    window.show()
    app.exec_()


