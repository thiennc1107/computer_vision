import cv2
import time
from ultis import *

adj_factor = 2.5
Known_distance = 30  # Inches
Known_width = 5.7  # Inche
ref_image = cv2.imread("ref_image/Ref_image.png")

ref_image_face_width, _, _, _ = face_data(ref_image)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)

def objectrecognition():
    result_all = []
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    while True:
        ret, image = cam.read()
        cv2.imwrite('opencv_frame.png', image)
        time.sleep(2)
        break

    cam.release()

    cv2.destroyAllWindows()

    img = cv2.imread('opencv_frame.png')
    thres = 0.45  # Threshold to detect object

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    distance_arr = []
    for bbox_i in bbox:
        distance = Distance_finder(
            Focal_length_found, Known_width, bbox_i[2]
        )
        distance_arr.append(int(distance) * adj_factor)
    cp = 0
    for classId, confidence, box, distance in zip(classIds.flatten(), confs.flatten(), bbox, distance_arr):
        if classNames[classId - 1] == 'person':
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite('opencv_frame_result.png', img)
            result = [confidence, distance]
            result_all.append(result)
            cp += 1
            
    if len(result_all) == 0:
        print("Không có ai trong khung hình")
    elif len(result_all) == 1:
        print(f"Có 1 người trong khung hình cách camera {result_all[0][1]} xăng-ti-mét")
    else:
        k_c = " "
        for i in range(0, len(result_all)):
            k_c += (str(result_all[i][1]) + ", ")
        print(f"Có {len(result_all)} người trong khung hình cách camera lần lượt là {k_c}" )
    return result_all

if __name__ == '__main__':
    result_all = objectrecognition()
    