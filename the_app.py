import recogniser as rg
import webcam as wb
import os
import cv2.cv2 as cv2

data_folder_path = "/mnt/54C615C6C615A8EE/my_projects/security_cam_web_app/face_recognizer"

subjects  = ["",]
person_counter = 1
person = []

new_entry = input("do you want to add new person:y or n").upper()
if new_entry == "Y":
    wb.new_person(subjects)




print("Preparing data...")
faces, labels = rg.prepare_training_data("train")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face = rg.FaceRecognizer()
face.train(faces,labels)

print("loading test image")
wb.take_attendance()
test_image = os.listdir(data_folder_path + '/test/')
print(test_image)
for x in test_image:
    test = cv2.imread(data_folder_path + '/test/'+x)
    print(x)

test1 =cv2.imread(data_folder_path+'/test/' + x)
predict = rg.predict(test1,subjects) #,labels)
cv2.imwrite('predict.jpg',predict)
print("Prediction complete")

rg.display_result(predict)