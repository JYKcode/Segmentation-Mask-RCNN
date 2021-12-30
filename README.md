# Project : Segmentation by Mask-RCNN
___
### 📣 Mask-RCNN을 통한 영역 분할 정리 및 구현 내용
- Mask-RCNN
    - 대표적인 객체 영역 분할 딥러닝 알고리즘
    - Faster R-CNN (object detection) + FCN (semantic segmentation)  
    
![image](https://user-images.githubusercontent.com/88880041/147714841-7867a7f3-acd0-440b-b411-0ea6fe084d9d.png)  
[참고]https://arxiv.org/pdf/1703.06870  
- Faster R-CNN을 통한 객체 검출 → bounding box information
- FCN을 통한 객체 단위 클래스 분류 → 객체 단위 mask map

### Structure
![image](https://user-images.githubusercontent.com/88880041/147714884-f7d438ef-85cf-4ba4-b763-9e41870911ef.png)  
[참고]https://towardsdatascience.com/computer-vision-instance-segmentation-with-mask-r-cnn-7983502fcad1
### Input
- Size: 임의의 크기 (auto resize)
- Scale: 1 (1 ~ 255)
- Mean: [0, 0, 0]
- RGB: True

### Output
- 2 output layers
    - 'detection_out_final' → boxes: boxes.shape=(1, 1, 100, 7)
    - 'detection_masks' → masks: masks.shape=(100, 90, 15, 15) 

### Mask-RCNN 모델 & 설정 파일
- 모델 : http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
- 설정 : https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbt
- 클래스 이름 : https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

### Result
![image](https://user-images.githubusercontent.com/88880041/147715120-b765267c-beeb-4c1a-ae3c-aa8777e9570b.png)
