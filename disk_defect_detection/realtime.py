#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
----------------------------------------------------------------------------
Created By  : Sertac İnce
Created Date: 21/07/2023
Updated Date: 25/07/2023
Version = 1.314
---------------------------------------------------------------------------
"""


import cv2
import tensorflow as tf
import numpy as np

# Loading model
model = tf.keras.models.load_model('modelRing.h5')

# Starting cam
cap = cv2.VideoCapture(0)
size = (500, 500)

class_labels = {1: 'OK', 0: 'Defective'}  

def preprocess_image(image):
    # Convert frames to grayscale (dataset was trained with grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    
    # Normalization
    processed_image = gray_image / 255.0
    
    # adding new dimension for single channel streaming
    processed_image = processed_image[..., np.newaxis]
    
    return processed_image

def postprocess_predictions(predictions, frame):    
    # Get predicted label
    predicted_label = class_labels[int(predictions>=0.5)]
    
    # Prediction probability 
    prob = predictions if predicted_label == 'Defective' else (1.0 - predictions)

    # Adding results to the frame
    result = cv2.putText(frame, f"Prediction: {predicted_label} ({prob:.2f})", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return result


def main():
    while True:
        # Getting frame
        ret, fframe = cap.read()
        # Resizing the frame
        frame = cv2.resize(fframe, size)  

        # Changing stream for prediction
        processed_frame = preprocess_image(frame)
        resized = np.expand_dims(processed_frame, axis=0)
        # Prediction
        [[predictions]] = model.predict(resized)
        
        result = postprocess_predictions(predictions, processed_frame)

        # Putting results to the screen
        cv2.imshow('Prediction', result)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main() 
    cap.release()
    cv2.destroyAllWindows()