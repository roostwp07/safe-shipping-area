from ultralytics import YOLO
import cv2

def detect_person(rtsp_url):
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, verbose=False)

        person_found = False

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    person_found = True
                    break
        
        if person_found:
            print("Person detected")

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()