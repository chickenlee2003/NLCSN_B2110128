import os
import pickle

import mediapipe as mp
import cv2
import numpy as np



mp_hands = mp.solutions.hands #chức năng liên quan đến nhận dạng và xử xử lý tay trong ảnh hoặc video
mp_drawing = mp.solutions.drawing_utils # Chức năng liên quan đến vẽ các điểm mốc
mp_drawing_styles = mp.solutions.drawing_styles #Chức năng liên quan đến kiểu vẽ các điểm mốc


#Nhận biết hình dạng và chuyển động của bàn tay
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data =[]
labels = []

for dir_ in os.listdir(DATA_DIR): #os.list.dir trả về một danh sách chứa các tên của các thực thể trong thư mục đã dc cho bởi path
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # ó.path.join: nối đường dẫn
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # chuyển hình ảnh thành dang rgb để nhập vào mediapie vì với mediapie
        #tất cả cá điểm phát hiện mốc luôn ở dạng rgb

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # lấy tất cả các hình ảnh, có một hình ảnh lấy i mang thông tin tất cả các điểm ảnh
            # multi_hand_landmarks là một danh sách chứa thông tin về bàn tay được phát  hiện trong ảnh hoặc video
            #hand_landmarks là thuộc tính chứa thông tin về các điểm landmark trên một tay cụ thể 
            #mỗi điểm là (x,y,z) là vị trí của diderm đó
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            a = np.array( results.multi_hand_landmarks)
            if(a.size==1):
                for i in range (21):
                    x=0
                    y=0
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(dir_)
    print(labels)    

#lưu file để train            
# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()