import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import keyboard  # For media control

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize System Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]
print(f"Volume Range: {min_vol} to {max_vol}")  # Debugging print

# Set Window Size to 640x480
window_width = 640
window_height = 480

# Define Button and Slider Positions
button_width, button_height = 100, 60  # Smaller buttons for smaller window
play_pause_button = (20, 20, 20 + button_width, 20 + button_height)
prev_button = (20, 100, 20 + button_width, 100 + button_height)
next_button = (20, 180, 20 + button_width, 180 + button_height)
vol_slider = (window_width - 250, 20, window_width - 50, 50)  # Horizontal slider at the top right
# Buffer distance for flexible detection
buffer_distance = 20  # Pixels around the buttons for flexible detection

# Track button states
button_states = {
    "play_pause": False,  # Fingers are not in the button area initially
    "prev": False,
    "next": False,
}

# Function to Draw UI Elements
def draw_ui(frame, current_vol):
    # Play/Pause Button (Black)
    cv2.rectangle(frame, (play_pause_button[0], play_pause_button[1]),
                  (play_pause_button[2], play_pause_button[3]), (0, 0, 0), -1)  # Black fill
    cv2.putText(frame, "Play/Pause", (play_pause_button[0] + 5, play_pause_button[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Smaller text

    # Previous Button (Black)
    cv2.rectangle(frame, (prev_button[0], prev_button[1]),
                  (prev_button[2], prev_button[3]), (0, 0, 0), -1)  # Black fill
    cv2.putText(frame, "Prev", (prev_button[0] + 30, prev_button[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Smaller text

    # Next Button (Black)
    cv2.rectangle(frame, (next_button[0], next_button[1]),
                  (next_button[2], next_button[3]), (0, 0, 0), -1)  # Black fill
    cv2.putText(frame, "Next", (next_button[0] + 30, next_button[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Smaller text

    # Volume Slider (Horizontal at the top right)
    cv2.rectangle(frame, (vol_slider[0], vol_slider[1]),
                  (vol_slider[2], vol_slider[3]), (0, 255, 0), 2)  # Green outline
    # Current Volume Indicator
    vol_x = int(np.interp(current_vol, [0, 100], [vol_slider[0], vol_slider[2]]))
    cv2.circle(frame, (vol_x, vol_slider[1] + 15), 10, (0, 0, 0), -1)  # Green circle for current volume
    # Display Current Volume Percentage
    vol_percentage = int(np.interp(current_vol, [min_vol, max_vol], [0, 100]))
    cv2.putText(frame, f"Vol: {round(current_vol)}%", (vol_slider[0], vol_slider[3] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Function to Check if Two Fingers are Up
def is_two_fingers_up(landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    return (index_tip.y < index_pip.y) and (middle_tip.y < middle_pip.y)

# Function to Handle Button Clicks with State Management
def handle_buttons(frame, landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * window_width), int(index_tip.y * window_height)  # Scaled to window size

    # Check if the fingertip is within the button area + buffer zone
def is_within_button(button, x, y):
        return (button[0] - buffer_distance < x < button[2] + buffer_distance and
                button[1] - buffer_distance < y < button[3] + buffer_distance)

    # Track button states
button_states = {
    "play_pause": False,  # Fingers are not in the button area initially
    "prev": False,
    "next": False,
}

# Function to Handle Button Clicks with State Management
def handle_buttons(frame, landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x, y = int(index_tip.x * window_width), int(index_tip.y * window_height)  # Scaled to window size

    # Check if the fingertip is within the button area + buffer zone
    def is_within_button(button, x, y):
        return (button[0] - buffer_distance < x < button[2] + buffer_distance and
                button[1] - buffer_distance < y < button[3] + buffer_distance)

    # Play/Pause Button
    if is_within_button(play_pause_button, x, y):
        if not button_states["play_pause"]:  # Fingers just entered the button area
            print("Play/Pause Clicked")
            keyboard.press_and_release('play/pause')  # Simulate Play/Pause key press
            button_states["play_pause"] = True  # Update state
    else:
        button_states["play_pause"] = False  # Reset state when fingers leave

    # Previous Button
    if is_within_button(prev_button, x, y):
        if not button_states["prev"]:  # Fingers just entered the button area
            print("Previous Clicked")
            keyboard.press_and_release('previous track')  # Simulate Previous Track key press
            button_states["prev"] = True  # Update state
    else:
        button_states["prev"] = False  # Reset state when fingers leave

    # Next Button
    if is_within_button(next_button, x, y):
        if not button_states["next"]:  # Fingers just entered the button area
            print("Next Clicked")
            keyboard.press_and_release('next track')  # Simulate Next Track key press
            button_states["next"] = True  # Update state
    else:
        button_states["next"] = False  # Reset state when fingers leave

# Function to Handle Volume Slider with Flexible Detection
def handle_volume_slider(frame, landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    x1, y1 = int(index_tip.x * window_width), int(index_tip.y * window_height)  # Scaled to window size
    x2, y2 = int(middle_tip.x * window_width), int(middle_tip.y * window_height)  # Scaled to window size

    # Debugging: Print fingertip positions
    print(f"Fingertip Positions: Index ({x1}, {y1}), Middle ({x2}, {y2})")

    # Check if fingertips are within the slider area + buffer zone
    if (vol_slider[0] - buffer_distance < x1 < vol_slider[2] + buffer_distance and
        vol_slider[0] - buffer_distance < x2 < vol_slider[2] + buffer_distance and 
        vol_slider[1] - buffer_distance < y1 < vol_slider[3] + buffer_distance and 
        vol_slider[1] - buffer_distance < y2 < vol_slider[3] + buffer_distance):
            avg_x = (x1 + x2) // 2  # Average X position of the fingertips
            vol = np.interp(avg_x, [vol_slider[0], vol_slider[2]], [min_vol, max_vol])
            vol_percentage = np.interp(avg_x, [vol_slider[0], vol_slider[2]], [0, 100])  # Map slider position to percentage
            volume.SetMasterVolumeLevelScalar(vol_percentage / 100, None)  # Set volume as a scalar (0.0 to 1.0)
            print(f"Volume set to: {int(vol)}%")  # Debugging print

# Main Loop
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Controlled Audio Player", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Controlled Audio Player", window_width, window_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect
    frame = cv2.resize(frame, (window_width, window_height))
    current_vol = volume.GetMasterVolumeLevelScalar() * 100  # Convert to percentage
    draw_ui(frame, current_vol)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_two_fingers_up(hand_landmarks):
                handle_buttons(frame, hand_landmarks)
                handle_volume_slider(frame, hand_landmarks)

    cv2.imshow("Hand Controlled Audio Player", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()