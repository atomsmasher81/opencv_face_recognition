import cv2.cv2 as cv2
import os

def create_dir(person_counter):
    path = "train/s{}".format(person_counter)

    try:
        os.mkdir(path)
    except OSError:
        if os.path.isdir(path):
            print("dir {} already there".format(path))
        else:
            print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def new_person(subject):
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    img_counter = 1
    subject.append(input("enter person name"))
    person_counter =len(subject)-1
    create_dir(person_counter)
    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)

        if c % 256 == 27:
              # ESC pressed
              print("Escape hit, closing...")
              break
        elif c % 256 == 32:
              # SPACE pressed
              img_name = "train/s{}/frame_{}.png".format(person_counter, img_counter)
              cv2.imwrite(img_name, frame)
              print("{} written!".format(img_name))
              img_counter += 1

    cap.release()
    cv2.destroyAllWindows()


def take_attendance():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    # img_counter = 1
    # subject.append(input("enter person name"))
    # person_counter =len(subject) -1
    # create_dir(person_counter)
    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(0)

        if c % 256 == 27:
              # ESC pressed
              print("Escape hit, closing...")
              break
        else:
              # any key pressed
              img_name = "test/frame_1.png"
              cv2.imwrite(img_name, frame)
              print("{} written!".format(img_name))


    cap.release()
    cv2.destroyAllWindows()




