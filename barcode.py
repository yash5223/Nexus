import cv2
import pandas as pd
from pyzbar.pyzbar import decode
from fuzzywuzzy import fuzz
import numpy as np
csv_file = 'complete_food_symptoms_dataset.csv'  
df = pd.read_csv(csv_file)
def search_in_csv(decoded_data):
    result = df[df.iloc[:, 0].astype(str).str.contains(decoded_data, na=False, case=False)]
    if not result.empty:
        return result.iloc[0]
    else:
        best_match = None
        highest_score = 0
        for index, row in df.iterrows():
            score = fuzz.ratio(decoded_data.lower(), str(row.iloc[0]).lower())
            if score > highest_score:
                highest_score = score
                best_match = row
        if best_match is not None:
            return best_match
        else:
            return None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()
result_found = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    decoded_objects = decode(frame)
    if decoded_objects and not result_found:
        for obj in decoded_objects:
            points = obj.polygon
            if len(points) == 4:
                pts = [tuple(pt) for pt in points]
                cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)
            else:
                cv2.circle(frame, (points[0].x, points[0].y), 5, (0, 0, 255), -1)
            barcode_data = obj.data.decode("utf-8")
            barcode_type = obj.type
            cv2.putText(frame, f"Barcode/QR: {barcode_data}", (obj.rect[0], obj.rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            search_result = search_in_csv(barcode_data)
            if search_result is not None:
                print("Found Data:")
                print(search_result)
                cv2.putText(frame, f"Found: {search_result.to_string()}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                result_found = True
                break
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or result_found:
        break
cap.release()
cv2.destroyAllWindows()
