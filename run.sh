nohup python -u main_moco.py /workspace/dataset/imageNet1k --resume ./checkpoint_0037.pth.tar -a resnet50 --lr 0.015 --batch-size 128 --dist-url tcp://localhost:10086 --world-size 1 --rank 0 --multiprocessing-distributed --moco-t 0.2 --mlp --cos --aug-plus > train_log2.log 2>&1 &