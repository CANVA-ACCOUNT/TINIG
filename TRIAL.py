import random
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("C:/Users/Naomi Irish/OneDrive/Documents/naomi code RAHHHHHHHHH/best.pt")

cap = cv2.VideoCapture(0)

detected_letters = []

previous_letter = None

alphabet = {i: chr(97 + i) for i in range(26)}

letter_colors = {}

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def draw_boxes(frame, boxes, confidences, names):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        class_id = int(box.cls[0])
        label = names[class_id]
        
        confidence = confidences[i].item()

        if confidence >= 0.7:
            
            if label not in letter_colors:
                letter_colors[label] = random_color()
            
            color = letter_colors[label]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{label} ({confidence*100:.2f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_detected_letters(letter_window, detected_letters):
    max_width = letter_window.shape[1] - 10
    max_height = letter_window.shape[0] - 10

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    title_y_position = 30

    cv2.putText(letter_window, "Detected Letters", (10, title_y_position), font, 1, (255, 255, 255), 2)

    x, y = 10, title_y_position + 40

    detected_text = ''.join(detected_letters)

    for letter in detected_letters:
        (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, font_thickness)

        if x + text_width > max_width:
            x = 10
            y += text_height + 10

        if y + text_height > max_height:
            break

        cv2.putText(letter_window, letter, (x, y), font, font_scale, (255, 255, 255), font_thickness)

        x += text_width + 7

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        names = result.names
        
        confidences = boxes.conf

        for i, box in enumerate(boxes):
            confidence = confidences[i].item()

            if confidence >= 0.7:
                class_id = int(box.cls[0])
                label = alphabet.get(class_id, None)

                if label != previous_letter:
                    detected_letters.append(label)
                    previous_letter = label

                    print(f"Detected letters: {detected_letters}")

                draw_boxes(frame, boxes, confidences, names)

    frame_height, frame_width = frame.shape[:2]

    letter_window_width = 300
    letter_window = np.zeros((frame_height, letter_window_width, 3), dtype=np.uint8)

    draw_detected_letters(letter_window, detected_letters)

    combined_frame = np.hstack((frame, letter_window))

    cv2.imshow("Sign Language Detection", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('x'):
        detected_letters.clear()
        previous_letter = None
        letter_colors.clear()
    elif key == 8:
        if detected_letters:
            detected_letters.pop()
            previous_letter = None

cap.release()
cv2.destroyAllWindows()
