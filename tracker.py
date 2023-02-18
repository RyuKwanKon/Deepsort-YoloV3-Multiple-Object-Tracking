import csv
from google.colab.patches import cv2_imshow
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort import preprocessing
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from absl import flags
import sys
from absl.flags import FLAGS
# 오류가 나면 실행
# flags.DEFINE_string("f", "", "kernel")
FLAGS(sys.argv)


# 오류 땜에 따로 가져옴
# from yolov3_tf2.utils import convert_boxes


def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int)
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2]-box[0])
        box[3] = int(box[3]-box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0, 0, 0, 0]:
            returned_boxes.append(box)
    return returned_boxes


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
# ./tools/generate_detections.py 84, 86번째 줄에 수정 net/%s:0 -> %s:0
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    'cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/testvideo.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results3.avi', codec,
                      vid_fps, (vid_width, vid_height))

current_frame = 0
file_path = './data/labels/text.csv'
#  file open
file = open(file_path, mode='w', newline='')
writer = csv.writer(file)

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    current_frame += 1

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    # imput 크기 조절 yolo 기본 크기 416 * 416
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []

    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])

    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(
        boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        # print(bbox[0], bbox[1], bbox[2], bbox[3])
        writer.writerow([current_frame, class_name+"-" +
                        str(track.track_id), bbox[0], bbox[1], bbox[2], bbox[3]])

        # 왼쪽 상단 x, y 오른쪽 하단 x, y, 색상, 두께
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), color, 2)
        # id 좌표
        cv2.rectangle(img,
                      (int(bbox[0]), int(bbox[1]-30)),
                      (int(bbox[0]) + (len(class_name) + len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img,
                    class_name+"-"+str(track.track_id),
                    (int(bbox[0]), int(bbox[1]-10)),
                    0, 0.75, (255, 255, 255), 2)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
    # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('output', 1920, 1080)
    # cv2.imshow('output', img)
    #  코랩에서 사용법
    cv2_imshow(img)

    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

file.close()
vid.release()
out.release()
cv2.destroyAllWindows()

# 초당 3포인트
