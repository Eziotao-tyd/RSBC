```bash
nohup python -u train.py --data_path=../dataset/data_relabelled.csv --num_classes=9 > ./checkpoints/test_1.log 2>&1 &
python calc.py --log_path=./checkpoints/test.log
python draw.py --pkl_path=./checkpoints/test.pkl --out_path=./test.png
```