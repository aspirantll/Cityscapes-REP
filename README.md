# instance segmentation based on boundary detection
This project implements a new approach for object segementation.
## requirements
The dependencies are as following:
* pytorch v1.3+
* torchvision v0.4.0+
* pycocotools v2.0.0
* torchsummary v1.5.1
* opencv-python v4.0.0+
* tqdm v4.43.0+
* sklearn
* Cython v0.29.15+
### install dependency
```shell script
pip install -r pip-requirements.txt
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
## train
The dataset only support the cityscapes, the train command as following:
```shell script
python train.py 
    --save_dir results/checkpoints  
    --train_dir /home/jovyan/work/val2017  
    --val_dir /home/jovyan/work/val2017 
    --batch_size 4  
    --checkpoint_span 100
```
