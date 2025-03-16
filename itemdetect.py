import cv2
import pytesseract
import numpy as np
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ingredients_list = ['sugar', 'flour', 'salt', 'egg', 'butter', 'milk', 'chocolate', 'vanilla', 'baking powder']
def extract_text_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    text = pytesseract.image_to_string(thresh_image)
    return text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
def find_ingredients_in_text(text, ingredients_list):
    found_ingredients = []
    for ingredient in ingredients_list:
        if ingredient in text:
            found_ingredients.append(ingredient)
    return found_ingredients
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('Live Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        captured_image = frame
        print("Image captured!")
        extracted_text = extract_text_from_image(captured_image)
        print(f"Extracted Text: {extracted_text}")
        cleaned_text = clean_text(extracted_text)
        print(f"Cleaned Text: {cleaned_text}")
        found_ingredients = find_ingredients_in_text(cleaned_text, ingredients_list)
        if found_ingredients:
            print("Found Ingredients:")
            for ingredient in found_ingredients:
                print(f"- {ingredient}")
        else:
            print("No ingredients found.")
        cv2.putText(captured_image, f"Found: {', '.join(found_ingredients)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captured Image with Ingredients', captured_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
