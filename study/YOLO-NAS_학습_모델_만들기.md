# 실행 환경
`python 3.10.12`, `google colab`

<br>

# 순서

### 1. custom data set 준비

([roboflow](https://public.roboflow.com/object-detection) 이용)

<img width="1712" alt="스크린샷 2024-04-02 오후 7 41 28" src="https://github.com/hongkikii/document/assets/110226866/c3a00f12-4ac2-4372-83bd-e325cfc479a4">

목록에서 원하는 data set을 찾아 클릭한다.

<br>

<img width="1722" alt="스크린샷 2024-04-02 오후 7 45 24" src="https://github.com/hongkikii/document/assets/110226866/faa1ed4a-efdf-4a61-9bf1-1f0a50a31872">

**raw**를 선택한다.

<br>

<img width="674" alt="스크린샷 2024-04-02 오후 5 54 31" src="https://github.com/hongkikii/document/assets/110226866/cc93802c-915b-42c1-a558-524f9ec3d143">

**Format**에서 알맞은 버전의 YOLO를 지정하고, **show download code**를 선택 후 **continue**를 클릭한다.

<br>

<img width="669" alt="스크린샷 2024-04-02 오후 5 55 02" src="https://github.com/hongkikii/document/assets/110226866/51a830f5-ca32-40cf-a32b-f5dd5845427f">

**Raw URL**에서 https 링크를 복사한다. (colab에서 사용하기 위함이다)

<br>

### 2. colab에서 실행하기

<img width="1716" alt="스크린샷 2024-04-02 오후 7 53 35" src="https://github.com/hongkikii/document/assets/110226866/38654ed4-62ec-4d8c-98de-fa396254ccf4">

```
!wget -O 파일지정이름 https경로
```
복사한 https 경로를 통해 파일을 다운로드 받고, 이름을 지정하여 저장한다.

<br>
<br>

<img width="1650" alt="스크린샷 2024-04-02 오후 7 53 43" src="https://github.com/hongkikii/document/assets/110226866/6b198fae-17e6-4d5b-9375-a810f12d3582">

```
import zipfile
with zipfile.ZipFile('압축파일경로') as target_file:
target_file.extractall('추출경로')
```
압축 파일을 해제하고, 해당 파일을 추출 경로에 저장한다.

```
!cat /추출경로/data.yaml
```
data.yaml 파일을 열어 설정을 확인한다.  
여기서 train, val, test는 별도로 지정할 예정이고, nc와 names는 그대로 사용할 것이다. 

<br>
<br>

<img width="1682" alt="스크린샷 2024-04-02 오후 7 53 50" src="https://github.com/hongkikii/document/assets/110226866/e2f21dc9-582e-4a11-acb7-7af1df81da3d">

```
!pip install PyYAML
```
colab에서 이용할 수 있게, yaml 파일을 커스텀해야 한다.  
이를 위해 PyYAML 라이브러리를 설치한다. 

```
import yaml

data = { 'train' : '/content/Uno_Data/train/images',
         'val' : '/content/Uno_Data/valid/images',
         'test' : '/content/Uno_Data/test/images',
         'names' : ['0', '1', '10', '11', '12', '13', '14', '2', '3', '4', '5', '6', '7', '8', '9'],
         'nc' : 15}

with open('/content/Uno_Data/Uno_Data.yaml', 'w') as f:
  yaml.dump(data, f)

with open('/content/Uno_Data/Uno_Data.yaml', 'r') as f:
  uno_yaml = yaml.safe_load(f)
  display(uno_yaml)
```

train, val, test의 경로를 절대경로로 직접 지정해주고, names와 nc는 그대로 가져온 뒤 `지정이름.yaml` 형태로 저장한다.

<br>
<br>

<img width="1475" alt="스크린샷 2024-04-02 오후 7 53 55" src="https://github.com/hongkikii/document/assets/110226866/77e7b29a-34b0-4057-98f0-bdc74b420dc2">

```
!pip install ultralytics
```
객체 탐지를 위해 ultralytics 라이브러리를 설치한다.

<br>
<br>

<img width="1654" alt="스크린샷 2024-04-02 오후 7 54 04" src="https://github.com/hongkikii/document/assets/110226866/1be5338c-2bf4-4c2d-855a-42326462ee37">

```
import ultralytics

ultralytics.checks()
```
라이브러리가 정상적으로 설치됐는지 체크한 후,

```
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
```

미리 학습된 YOLOv8n 모델의 가중치를 이용해 모델을 초기화한다.

<br>
<br>

<img width="1666" alt="스크린샷 2024-04-02 오후 7 05 02" src="https://github.com/hongkikii/document/assets/110226866/34c3e246-2621-4c00-a0f6-e9a9327046cc">

```
model.train(data='yaml파일경로', epochs=100, patience=30, batch=32, imgsz=416)
```

모델을 실제로 트레이닝한다. 이 과정에서 파라미터 값들은 조정해나가야 한다.

> epochs : 학습 반복 횟수

> patience : 검증 손실이 개선되지 않는 에폭 수, 예를 들어 위 코드에서 30 에폭 동안 검증 손실이 개선되지 않으면 학습을 조기 종료한다.

> batch : 한 번의 학습(batch)에 사용할 데이터 수, 예를 들어 위 코드에서 32로 지정 시 한 번의 학습에 16개의 데이터 샘플을 사용하여 가중치를 업데이트.

> imgsz : 이미지의 크기

<br>
<br>


