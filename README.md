pip install -r requirements.txt

├───data
│   ├───Images
│   └───Lists
├───outputs
└───src

python src/train.py --model mobilenet_v3_large --fold 0
python src/train.py --model mobilenet_v3_large --fold 1
python src/train.py --model mobilenet_v3_large --fold 2
python src/train.py --model mobilenet_v3_large --fold 3
python src/train.py --model mobilenet_v3_large --fold 4

python src/train.py --model efficientnet_b0 --fold 0
python src/train.py --model efficientnet_b0 --fold 1
python src/train.py --model efficientnet_b0 --fold 2
python src/train.py --model efficientnet_b0 --fold 3
python src/train.py --model efficientnet_b0 --fold 4

python src/train.py --model resnet50 --fold 0
python src/train.py --model resnet50 --fold 1
python src/train.py --model resnet50 --fold 2
python src/train.py --model resnet50 --fold 3
python src/train.py --model resnet50 --fold 4