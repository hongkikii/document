# 실행 환경
`python 3.10.12`, `google colab`

<br>

# 순서

### 1. super-gradients 라이브러리 다운로드

```
!pip install super-gradients
```


### 2. 모델 생성을 위한 설정 수행

```
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm

import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
```

### 3. 파일을 다운로드하여 시스템에 저장

```
def download_file(url, save_name):
  if not os.path.exists(save_name):
    print("Downloading file")
    file = requests.get(url, stream=True)
    total_size = int(file.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size,
        unit='18',
        unit_scale=True
    )
    with open(os.path.join(save_name), 'wb') as f:
      for data in file.iter_content(block_size):
        progress_bar.update(len(data))
        f.write(data)
    progress_bar.close()
  else:
    print('File already present')

download_file(
    'https://www.dropbox.com/s/xc2890eh8ujy3cu/hituav-a-highaltitude-infrared-thermal-dataset.zip?dl=1',
    'hituav-a-highaltitude-infrared-thermal-dataset.zip'
)
```

### 4. zip 파일 압축 해제

```
def unzip(zip_file=None):
  try:
    with zipfile.ZipFile(zip_file) as z:
      z.extractall("./")
      print("Extracted all")
  except:
    print("Invalid file")

unzip('hituav-a-highaltitude-infrared-thermal-dataset.zip')
```

### 5. 훈련에 사용할 데이터셋의 디렉토리 경로와 클래스 정보 설정

```
ROOT_DIR = 'hit-uav'
train_imgs_dir = 'images/train'
train_labels_dir = 'labels/train'
val_imgs_dir = 'images/val'
val_labels_dir = 'labels/val'
test_imgs_dir = 'images/test'
test_labels_dir = 'labels/test'
classes = ['Person', 'Car', 'Bicycle', 'OtherVechicle', 'DontCare']
```

### 6. dataset_params 정의

```
dataset_params = {
    'data_dir' : ROOT_DIR,
    'train_images_dir' : train_imgs_dir,
    'train_labels_dir' : train_labels_dir,
    'val_images_dir' : val_imgs_dir,
    'val_labels_dir' : val_labels_dir,
    'test_images_dir' : test_imgs_dir,
    'test_labels_dir' : test_labels_dir,
    'classes' : classes
}
```

### 7. 주요 매개변수 설정

```
EPOCHS = 50
BATCH_SIZE = 16
WORKERS = 8
```

### 8. 각 클래스에 대한 무작위 색상 생성

```
colors = np.random.uniform(0, 255, size=(len(classes), 3))
```

### 9. YOLO 바운딩 박스 포맷을 일반적인 바운딩 박스 포맷으로 변환

```
def yolo2bbox(bboxes):
  xmin, ymin = bboxes[0] - bboxes[2]/2, bboxes[1]-bboxes[3]/2
  xmax, ymax = bboxes[0] + bboxes[2]/2, bboxes[1]+bboxes[3]/2
  return xmin, ymin, xmax, ymax
```

### 10. 단일 이미지와 레이블 시각화

```
def plot_box(image, bboxes, labels):
  height, width, _ = image.shape
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  tf = max(lw - 1, 1)
  for box_num, box in enumerate(bboxes):
    x1, y1, x2, y2 = yolo2bbox(box)
    xmin = int(x1*width)
    ymin = int(y1*height)
    xmax = int(x2*width)
    ymax = int(y2*height)

    p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))

    class_name = classes[int(labels[box_num])]

    color=colors[classes.index(class_name)]

    cv2.rectangle(
        image,
        p1, p2,
        color=color,
        thickness=lw,
        lineType=cv2.LINE_AA
    )

    w, h = cv2.getTextSize (
        class_name,
        0,
        fontScale=lw / 3,
        thickness=tf
    )[0]

    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

    cv2.rectangle(
        image,
        p1, p2,
        color = color,
        thickness = -1,
        lineType=cv2.LINE_AA
    )

    cv2.putText(
        image,
        class_name,
        (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=lw/3.5,
        color=(255, 255, 255),
        thickness=tf,
        lineType=cv2.LINE_AA
    )
  return image
```

### 11. 복수 이미지와 레이블 시각화

```
def plot(image_path, label_path, num_samples):
  all_training_images = glob.glob(image_path+'/*')
  all_training_labels = glob.glob(label_path+'/*')
  all_training_images.sort()
  all_training_labels.sort()

  temp = list(zip(all_training_images, all_training_labels))
  random.shuffle(temp)
  all_training_images, all_training_labels = zip(*temp)
  all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)

  num_images = len(all_training_images)

  if num_samples == -1:
    num_samples = num_images

  plt.figure(figsize=(15, 12))
  for i in range(num_samples):
    image_name = all_training_images[i].split(os.path.sep)[-1]
    image = cv2.imread(all_training_images[i])
    with open(all_training_labels[i], 'r') as f:
      bboxes = []
      labels = []
      label_lines = f.readlines()
      for label_line in label_lines:
        label, x_c, y_c, w, h = label_line.split(' ')
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)
        bboxes.append([x_c, y_c, w, h])
        labels.append(label)
    result_image = plot_box(image, bboxes, labels)
    plt.subplot(2, 2, i+1)
    plt.imshow(image[:, :, ::-1])
    plt.axis('off')
  plt.tight_layout()
  plt.show()
```

### 12. plot 함수를 호출하여 시각화 작업 수행

```
plot(
    image_path=os.path.join(ROOT_DIR, train_imgs_dir),
    label_path=os.path.join(ROOT_DIR, train_labels_dir),
    num_samples=4,
)
```
![image](https://github.com/hongkikii/document/assets/110226866/5ae223e9-b8f2-4ab2-97d4-b8ed70a1ff96)


### 13. training & valdiation data 로드, 준비

```
train_data = coco_detection_yolo_format_train(
    dataset_params ={
        'data_dir' : dataset_params['data_dir'],
        'images_dir' : dataset_params['train_images_dir'],
        'labels_dir' : dataset_params['train_labels_dir'],
        'classes' : dataset_params['classes']
    },
    dataloader_params = {
        'batch_size' : BATCH_SIZE,
        'num_workers': WORKERS
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir' : dataset_params['data_dir'],
        'images_dir' : dataset_params['val_images_dir'],
        'labels_dir' : dataset_params['val_labels_dir'],
        'classes' : dataset_params['classes']
    },
    dataloader_params = {
        'batch_size' : BATCH_SIZE,
        'num_workers': WORKERS
    }
)
```

### 14. train_data 데이터셋 객체에 적용된 변환(transforms) 목록 조회

```
train_data.dataset.transforms
```

<br>

transforms[0], transforms.pop(2)의 형태로도 이용 가능

### 15. 변환(transformations)을 시각적으로 표시

```
train_data.dataset.plot(plot_transformed_data=True)
```
![image](https://github.com/hongkikii/document/assets/110226866/8310958b-bc60-4e66-bd94-f87265f7d892)


### 16. train_params 정의

```
train_params = {
    'silent_mode' : False,
    "average_best_models" : True,
    "warmup_mode" : "linear_epoch_step",
    "warmup_initial_lr" : 1e-6,
    "lr_warmup_epochs" : 3,
    "initial_lr": 5e-4,
    "lr_mode" : "cosine",
    "cosine_final_lr_ratio" : 0.1,
    "optimizer" : "Adam",
    "optimizer_params" : {"weight_decay" : 0.0001},
    "zero_weight_decay_on_bias_and_bn" : True,
    "ema" : True,
    "ema_params" : {"decay": 0.9, "decay_type" : "threshold"},
    "max_epochs" : EPOCHS,
    "mixed_precision" : True,
    "loss" : PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch" : 'mAP@0.50:0.95'
}
```

### 17. 훈련하려는 모델 설정

```
models_to_train = [
    'yolo_nas_s',
    'yolo_nas_m',
    'yolo_nas_l'
]
```

### 18. 체크포인트 파일 변수 할당

```
CHECKPOINT_DIR = 'checkpoints'
```

### 19. 모델 훈련

```
for model_to_train in models_to_train:
  trainer = Trainer(
      experiment_name=model_to_train,
      ckpt_root_dir=CHECKPOINT_DIR
  )

  model = models.get(
      model_to_train,
      num_classes=len(dataset_params['classes']),
      pretrained_weights="coco"
  )

  trainer.train(
      model=model,
      training_params=train_params,
      train_loader=train_data,
      valid_loader=val_data
  )
```

[2024-04-09 09:32:31] INFO - sg_trainer_utils.py - TRAINING PARAMETERS:  
    - Mode:                         Single GPU  
    - Number of GPUs:               0          (0 available on the machine)  
    - Full dataset size:            2008       (len(train_set))  
    - Batch size per GPU:           4          (batch_size)  
    - Batch Accumulate:             1          (batch_accumulate)  
    - Total batch size:             4          (num_gpus * batch_size)  
    - Effective Batch size:         4          (num_gpus * batch_size * batch_accumulate)  
    - Iterations per epoch:         502        (len(train_loader))  
    - Gradient updates per epoch:   502        (len(train_loader) / batch_accumulate)  
    - Model: YoloNAS_S  (19.02M parameters, 19.02M optimized)  
    - Learning Rates and Weight Decays:  
      - default: (19.02M parameters). LR: 0.0005 (19.02M parameters) WD: 0.0, (42.14K parameters), WD: 0.0001, (18.98M parameters)  


Train epoch 0: 100%|██████████| 502/502 [2:13:29<00:00, 15.96s/it, PPYoloELoss/loss=3.63, PPYoloELoss/loss_cls=2.17, PPYoloELoss/loss_dfl=0.572, PPYoloELoss/loss_iou=0.887, gpu_mem=0]  
Validating: 100%|██████████| 72/72 [05:21<00:00,  4.47s/it]  
[2024-04-09 11:51:29] INFO - base_sg_logger.py - Checkpoint saved in checkpoints/yolo_nas_s/RUN_20240409_093229_275059/ckpt_best.pth  
[2024-04-09 11:51:29] INFO - sg_trainer.py - Best checkpoint overriden: validation mAP@0.50:0.95: 0.0022039629984647036  



SUMMARY OF EPOCH 0  
├── Train  
│   ├── Ppyoloeloss/loss_cls = 2.1737  
│   ├── Ppyoloeloss/loss_iou = 0.8875  
│   ├── Ppyoloeloss/loss_dfl = 0.5718  
│   └── Ppyoloeloss/loss = 3.633  
└── Validation  
    ├── Ppyoloeloss/loss_cls = 2.0794  
    ├── Ppyoloeloss/loss_iou = 0.7525  
    ├── Ppyoloeloss/loss_dfl = 0.4567  
    ├── Ppyoloeloss/loss = 3.2886  
    ├── Precision@0.50 = 0.0573  
    ├── Recall@0.50 = 0.0118  
    ├── Map@0.50 = 0.0054  
    ├── F1@0.50 = 0.012  
    ├── Best_score_threshold = 0.07  
    ├── Precision@0.50:0.95 = 0.0329  
    ├── Recall@0.50:0.95 = 0.0055  
    ├── Map@0.50:0.95 = 0.0022  
    └── F1@0.50:0.95 = 0.0057  


Train epoch 1: 100%|██████████| 502/502 [2:15:31<00:00, 16.20s/it, PPYoloELoss/loss=2.18, PPYoloELoss/loss_cls=1.12, PPYoloELoss/loss_dfl=0.426, PPYoloELoss/loss_iou=0.64, gpu_mem=0]  
Validating epoch 1: 100%|██████████| 72/72 [05:59<00:00,  4.99s/it]  
[2024-04-09 14:13:07] INFO - base_sg_logger.py - Checkpoint saved in checkpoints/yolo_nas_s/RUN_20240409_093229_275059/ckpt_best.pth  
[2024-04-09 14:13:07] INFO - sg_trainer.py - Best checkpoint overriden: validation mAP@0.50:0.95: 0.17470310628414154  


SUMMARY OF EPOCH 1  
├── Train  
│   ├── Ppyoloeloss/loss_cls = 1.1161  
│   │   ├── Epoch N-1      = 2.1737 (↘ -1.0577)  
│   │   └── Best until now = 2.1737 (↘ -1.0577)  
│   ├── Ppyoloeloss/loss_iou = 0.6396  
│   │   ├── Epoch N-1      = 0.8875 (↘ -0.2479)  
│   │   └── Best until now = 0.8875 (↘ -0.2479)    
│   ├── Ppyoloeloss/loss_dfl = 0.4256  
│   │   ├── Epoch N-1      = 0.5718 (↘ -0.1461)  
│   │   └── Best until now = 0.5718 (↘ -0.1461)  
│   └── Ppyoloeloss/loss = 2.1813  
│       ├── Epoch N-1      = 3.633  (↘ -1.4517)  
│       └── Best until now = 3.633  (↘ -1.4517)  
└── Validation  
    ├── Ppyoloeloss/loss_cls = 0.9507  
    │   ├── Epoch N-1      = 2.0794 (↘ -1.1286)  
    │   └── Best until now = 2.0794 (↘ -1.1286)  
    ├── Ppyoloeloss/loss_iou = 0.578  
    │   ├── Epoch N-1      = 0.7525 (↘ -0.1746)  
    │   └── Best until now = 0.7525 (↘ -0.1746)  
    ├── Ppyoloeloss/loss_dfl = 0.3971  
    │   ├── Epoch N-1      = 0.4567 (↘ -0.0596)  
    │   └── Best until now = 0.4567 (↘ -0.0596)  
    ├── Ppyoloeloss/loss = 1.9258  
    │   ├── Epoch N-1      = 3.2886 (↘ -1.3628)  
    │   └── Best until now = 3.2886 (↘ -1.3628)  
    ├── Precision@0.50 = 0.0436  
    │   ├── Epoch N-1      = 0.0573 (↘ -0.0137)  
    │   └── Best until now = 0.0573 (↘ -0.0137)  
    ├── Recall@0.50 = 0.6603  
    │   ├── Epoch N-1      = 0.0118 (↗ 0.6486)  
    │   └── Best until now = 0.0118 (↗ 0.6486)  
    ├── Map@0.50 = 0.3706  
    │   ├── Epoch N-1      = 0.0054 (↗ 0.3652)  
    │   └── Best until now = 0.0054 (↗ 0.3652)  
    ├── F1@0.50 = 0.0804  
    │   ├── Epoch N-1      = 0.012  (↗ 0.0684)  
    │   └── Best until now = 0.012  (↗ 0.0684)  
    ├── Best_score_threshold = 0.33  
    │   ├── Epoch N-1      = 0.07   (↗ 0.26)  
    │   └── Best until now = 0.07   (↗ 0.26)  
    ├── Precision@0.50:0.95 = 0.0245  
    │   ├── Epoch N-1      = 0.0329 (↘ -0.0085)  
    │   └── Best until now = 0.0329 (↘ -0.0085)  
    ├── Recall@0.50:0.95 = 0.363  
    │   ├── Epoch N-1      = 0.0055 (↗ 0.3575)  
    │   └── Best until now = 0.0055 (↗ 0.3575)  
    ├── Map@0.50:0.95 = 0.1747  
    │   ├── Epoch N-1      = 0.0022 (↗ 0.1725)  
    │   └── Best until now = 0.0022 (↗ 0.1725)  
    └── F1@0.50:0.95 = 0.0451  
        ├── Epoch N-1      = 0.0057 (↗ 0.0395)  
        └── Best until now = 0.0057 (↗ 0.0395)  

### 20. 평가 그래프 그리기
loss, Precision-Recall Curve, mAP, Approximate Average Precision Curve  
(epochs 2 기준)

```
import matplotlib.pyplot as plt

epochs = [0, 1]

train_loss = [3.633, 2.1813]
val_loss = [3.2886, 1.9258]

map_50 = [0.0054, 0.3706]
map_50_95 = [0.0022, 0.1747]

precision_50 = [0.0573, 0.0436]
recall_50 = [0.0118, 0.6603]
precision_50_95 = [0.0329, 0.0245]
recall_50_95 = [0.0055, 0.363]


plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(recall_50, precision_50, label='Precision@0.50')
plt.plot(recall_50_95, precision_50_95, label='Precision@0.50:0.95')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, map_50, label='mAP@0.50')
plt.plot(epochs, map_50_95, label='mAP@0.50:0.95', color='green')
plt.title('Mean Average Precision (mAP)')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(recall_50, precision_50, label='Precision@0.50-Recall Curve', color='blue')
plt.fill_between(recall_50, precision_50, step='post', alpha=0.2, color='blue')
plt.plot(recall_50_95, precision_50_95, label='Precision@0.50:0.95-Recall Curve', color='green')
plt.fill_between(recall_50_95, precision_50_95, step='post', alpha=0.2, color='green')
plt.title('Approximate Average Precision Curves')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
```
![image](https://github.com/hongkikii/document/assets/110226866/2d07356b-5d0a-4b75-af70-1a3b0ce8d20b)
![image](https://github.com/hongkikii/document/assets/110226866/d15e6ff2-f7b1-432c-b09c-f6c2cd20f95a)
![image](https://github.com/hongkikii/document/assets/110226866/453f7454-365d-4b55-98ef-ba20c7859d6d)
![image](https://github.com/hongkikii/document/assets/110226866/0460dc80-be54-4bc7-b7ff-9406c8c7d728)




### 참고
https://youtu.be/vfQYRJ1x4Qg
