# yuctian-representation-flow-cvpr19-paddle

#### 1.训练
python main.py -phase train -size 224 -length 24 -batch_size 24 -save_dir ./model

#### 2.测试
python main.py -phase eval -size 224 -length 32 -batch_size 32 -model your_model_path
