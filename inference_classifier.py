import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Toi', 1: 'Ten', 2: 'Ban',
                3: 'Yes', 4: 'hoan thanh',
                  5: 'quen', 6: 'xin chao', 7: 'giup do',
                    8: 'thich', 9: 'khong', 10: 'cam on',
               11: 'ban', 12: 'vui ve', 13: 'buon',
                 14: 'dien thoai', 15: 'K', 16: 'I',
                   17: 'E', 18: 'T', 19: 'tam biet'}
while True:

    data_aux = []
    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            a=np.array(results.multi_hand_landmarks) 
            if (a.size==1):
                for i in range (21):
                    x=0
                    y=0
                    data_aux.append(x)
                    data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.putText(frame, predicted_character,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 5,cv2.LINE_AA)

    cv2.imshow('Cửa sổ nhận diện', frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()