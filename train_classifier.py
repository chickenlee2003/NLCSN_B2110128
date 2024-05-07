import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

#Chuyển đổi dữ liệu thành dạng nparray
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#tách tập dữ liệu thành train test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle= True, stratify=labels)

#sử dụng rừng cây quyết định để train model
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)


# Kiểm tra độ chính xác
score = accuracy_score(y_predict, y_test)

print('{}% of sample were classified corect'.format(score*98.6))

#Lưu trữ mô hình huấn luyện
# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()


