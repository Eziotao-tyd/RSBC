# Usage

## 1、prepare data

```bash
# relabel data to start from 0
python ./dataset/data_change_label.py --data_path=./dataset/data.csv
```

## 2、train & test

```bash
# default configs in ./utils/config.py
nohup python -u train.py --data_path=../dataset/data_relabelled.csv --num_classes=9 > ./checkpoints/test_1.log 2>&1 &
nohup python -u train.py --data_path=../dataset/data_2_relabelled.csv --num_classes=20 > ./checkpoints/test_2.log 2>&1 &
```

## 3、draw result curve

```bash
python calc.py --log_path=./checkpoints/test_1.log
python draw.py --pkl_path=./checkpoints/test_1.pkl --out_path=./test_1.png
```