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
본 문서에서는 uno 카드에 나타난 숫자 데이터를 이용한다.

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

> batch : 한 번의 학습(batch)에 사용할 데이터 수, 예를 들어 위 코드처럼 32로 지정 시 한 번의 학습에 32개의 데이터 샘플을 사용하여 가중치를 업데이트한다.

> imgsz : 이미지의 크기

<br>

(epochs 3 기준 결과)

<img width="1608" alt="스크린샷 2024-04-02 오후 9 36 37" src="https://github.com/hongkikii/document/assets/110226866/eb24b475-59a5-4b7b-a894-efdef3c8ddf5">
<img width="1611" alt="스크린샷 2024-04-02 오후 9 36 49" src="https://github.com/hongkikii/document/assets/110226866/0a11f85b-f912-4e57-bd70-54517c77fa77">
<img width="1606" alt="스크린샷 2024-04-02 오후 9 37 54" src="https://github.com/hongkikii/document/assets/110226866/7cc884c3-07f2-4faf-a111-921f70f7ea8d">


<br>
<br>


<img width="1653" alt="스크린샷 2024-04-02 오후 9 42 06" src="https://github.com/hongkikii/document/assets/110226866/b9228785-6617-45fc-9a7f-37219f5fe24b">
<img width="1644" alt="스크린샷 2024-04-02 오후 9 42 19" src="https://github.com/hongkikii/document/assets/110226866/4093f8f4-fe34-4c48-9dea-7bd59886b9e9">

```
result = model.predict(source='테스트데이터경로', save=True)
```

테스트 데이터를 이용해 모델을 테스트하고, 결과를 저장한다.

<br>
<br>

<img width="1656" alt="스크린샷 2024-04-02 오후 9 45 04" src="https://github.com/hongkikii/document/assets/110226866/3cee3c4b-ee6a-41fd-b751-4d011fa61509">

```
from IPython.display import Image, display
import os

result_dir = "이미지저장경로"
image_files = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]

for image_file in image_files:
    display(Image(os.path.join(result_dir, image_file)))
```
모델 테스트 시 **Results saved to**를 통해 결과가 저장된 경로를 확인하고, 해당 경로를 이용해 저장한 이미지를 출력한다.  

<br>

(결과 이미지)
<br>
<img width="414" alt="스크린샷 2024-04-02 오후 9 45 22" src="https://github.com/hongkikii/document/assets/110226866/ad23228f-55c2-4116-9b23-5d56e6b76425">
<img width="415" alt="스크린샷 2024-04-02 오후 9 45 30" src="https://github.com/hongkikii/document/assets/110226866/4c7f2ddc-a16b-415b-b4b7-618cc599013d">
<img width="413" alt="스크린샷 2024-04-02 오후 9 45 37" src="https://github.com/hongkikii/document/assets/110226866/bcdb4dc5-1a04-4b8c-8eea-7d6bba72b6d2">
<img width="412" alt="스크린샷 2024-04-02 오후 9 46 32" src="https://github.com/hongkikii/document/assets/110226866/bc5254f6-d18e-48f1-8155-08439239fd96">
<img width="412" alt="스크린샷 2024-04-02 오후 9 46 39" src="https://github.com/hongkikii/document/assets/110226866/2fb1f276-55ae-4b6c-8839-7ae6ff252a41">
<img width="412" alt="스크린샷 2024-04-02 오후 9 46 47" src="https://github.com/hongkikii/document/assets/110226866/ac2bbdc0-ab1e-4dde-bd6c-63a7f8378f29">
<img width="413" alt="스크린샷 2024-04-02 오후 9 47 09" src="https://github.com/hongkikii/document/assets/110226866/5c23db1e-1fd3-4f4c-9a60-3cb571b5fbe3">
<img width="410" alt="스크린샷 2024-04-02 오후 10 07 19" src="https://github.com/hongkikii/document/assets/110226866/43363d9e-c5d0-495f-90a7-ce0dc70bcae9">
