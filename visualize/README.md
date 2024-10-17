# 실행 방법
1. test 데이터에 predict 하는 것처럼 validation data에 대해 predict
2. csv 파일 저장 후 진행
## 실행하기 전
기존 inference.ipynb 코드에 있는
```python
for i, out in enumerate(output):
    prediction_string = ''
        continue
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
```
에서
```python
    if i not in img_ids:
        continue
```
추가해야 함
```python
for i, out in enumerate(output):
    prediction_string = ''
    # 추가한 부분
    if i not in img_ids:
        continue
    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
```
### 폴더 세팅 예시

```
dataset/
│
├── train/
│   ├── 0000.jpg
│   └── ... 
├── train_split.json
└── val_split.json

result/                         
│
└── submission.csv
```

### submission.csv 예시
                    
PredictionString  | image_id
------------- | -------------
label score xmin ymin xmax ymax ...  | test/0000.jpg
label score xmin ymin xmax ymax ...  | test/0006.jpg

### 시각화 예시
![bbox](https://github.com/user-attachments/assets/b5469f47-d207-4af6-8524-fecfc7a95ca9)