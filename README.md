# Deepsort-YoloV3-Multiple-Object-Tracking

## MOT좌표 구하기 공부

### 공부 자료
- github: https://github.com/emasterclassacademy/Single-Multiple-Custom-Object-Detection-and-Tracking
- youtube: https://www.youtube.com/watch?v=zi-62z-3c4U&t=108s

### pip
```
!pip install absl-py 
!pip install yolov3-tf2
!pip3 install deep-sort-realtime
```

### 실행
1. YoloV3 weights를 ./weights에 다운로드 (references)
2. convert.py파일을 실행
3. detection영상 ./data/video에 다운로드
4. tracker 실행

![results3_AdobeExpress](https://user-images.githubusercontent.com/97783148/219853701-f54ba73f-7a20-4b61-b4d5-7c94f189cbf0.gif)

### references
- Yolov3 weights: https://www.youtube.com/redirectevent=video_description&redir_token=QUFFLUhqbmFIbFZYSjZBRzgzRGNGZ0ZIaHdfMUlzcng2UXxBQ3Jtc0tta3hoclRtb0prY2VMVEY4MkhwazVKU0JieXZlR1MxeVZ1a25sOGxGcHNrTmVGb3U3YzRLTkk3cWo3d0prUDkzU0NuZUZ6dDNCZ1FOQTBwaTJNaTZNeHdnYmkyY1FXRGdhVzRtTXZxU1pjOHVmdEdyTQ&q=https%3A%2F%2Fpjreddie.com%2Fmedia%2Ffiles%2Fyolov3.weights&v=zi-62z-3c4U
