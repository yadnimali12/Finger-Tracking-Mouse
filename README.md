Description

Finger Tracking Mouse is a Python-based touchless virtual mouse that uses hand gestures to control computer operations. It enables users to move the cursor and perform mouse actions without using a physical mouse.

Technologies Used

Python – Core programming
OpenCV – Webcam access and real-time video processing
MediaPipe – Hand and finger landmark detection
PyAutoGUI – Cursor movement and mouse actions
NumPy – Coordinate and numerical processing

How It Works??

Webcam Capture: The webcam continuously captures live video of the user's hand.
Hand Detection: MediaPipe detects the hand and identifies key landmarks such as fingertips and joints.
Finger Tracking: The system tracks the position and movement of the fingers in real time.
Gesture Recognition: Different finger positions and movements are interpreted as specific mouse gestures.
Mouse Control: PyAutoGUI converts these gestures into actions such as cursor movement, left/right click, double-click, drag-and-drop, and scrolling.
Smoothing: A filtering mechanism reduces unwanted hand movement and cursor jitter, making the interaction smoother and more stable.
Visual Feedback: The application displays the detected hand and tracking information through a real-time overlay.
