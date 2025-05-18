import tkinter as tk
from tkinter import filedialog, Label
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading

# Load the YOLOv8 model
model_path = "C:/Users/Muneeb Azad/Desktop/pro/best.pt"
model = YOLO(model_path)

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection in Video")
        self.root.geometry("800x600")

        self.label = Label(root)
        self.label.pack()

        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack()

    def upload_video(self):
        video_path = filedialog.askopenfilename()
        if video_path:
            threading.Thread(target=self.process_video, args=(video_path,)).start()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run the model on the frame
            results = model(frame)

            # Draw bounding boxes and labels on the frame
            for result in results:
                for bbox in result.boxes:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    label = model.names[int(bbox.cls[0])]
                    confidence = bbox.conf[0]
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Display the frame
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            # Update the GUI
            self.root.update_idletasks()

        cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()