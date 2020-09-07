import os
import argparse
import numpy as np
import paddle.fluid as fluid
import time
import json 
import reader
import resnet_2d_model as MODEL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='rgb',help='rgb or flow')
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-length', type=int, default=24)
    parser.add_argument('-learnable', type=str, default='[0,1,1,1,1]')
    parser.add_argument('-use_gpu', type=int,default=1)
    parser.add_argument('-pretrain', type=str,default='./model/latest')
    parser.add_argument('-save_dir', type=str,default='./model')
    parser.add_argument('-epoch', type=int,default=200)
    parser.add_argument('-epoch_num', type=int,default=0)
    parser.add_argument('-phase', type=str,default='train')
    parser.add_argument('-model', type=str,default='./model/best')
    parser.add_argument('-size', type=int,default=224)
    parser.add_argument('-dataset',type=str,default='hmdb',help ='ucf or hmdb')
    parser.add_argument('-train_file', type=str,default='new_train.txt')
    parser.add_argument('-test_file', type=str,default='new_test.txt')
    args = parser.parse_args()
    return args
def eval(args):
    place = fluid.CPUPlace() if not args.use_gpu else fluid.CUDAPlace(0) 
   
    with fluid.dygraph.guard(place):
      
        eval_model =MODEL.resnet_2d_v1(50,51,size=args.size,batch_size=args.batch_size)
        
        para_state_dict, _ = fluid.load_dygraph(args.model)
        eval_model.load_dict(para_state_dict)
        eval_model.eval()
        eval_reader = reader.DS(args.test_file, 'data/data47656/hmdb/',random=False, model='2d', mode=args.mode,length=args.length, batch_size=args.batch_size,size=args.size).create_reader()
       
        acc_list = []
        for batch_id, data in enumerate(eval_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
   
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            out = eval_model(img)
            acc = fluid.layers.accuracy(input=out, label=label)
            acc_list.append(acc.numpy()) 
            print(batch_id,'准确率:', acc.numpy())

        print("测试集准确率为:{}".format(np.mean(acc_list)))

def train(args):
    
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        

        train_model =MODEL.resnet_2d_v1(50,51,size=args.size,batch_size=args.batch_size)
          
        clip = fluid.clip.GradientClipByNorm(clip_norm=1.0)
        opt = fluid.optimizer.AdamOptimizer(0.001, 0.9,parameter_list=train_model.parameters(), regularization=fluid.regularizer.L2Decay(
        regularization_coeff=1e-6),epsilon=1e-8)#,grad_clip=clip)
        for i in train_model.state_dict():
            print(i+' '+str(train_model.state_dict()[i].shape))
        if args.pretrain:
            
            model, _ = fluid.dygraph.load_dygraph(args.pretrain)
            train_model.load_dict(model)
        train_model.train()
        # build model
        if not os.path.exists(args.save_dir):
             os.makedirs(args.save_dir)

        # get reader
      
        train_reader =  reader.DS(args.train_file, 'data/data47656/hmdb/', model='2d', mode=args.mode, length=args.length, batch_size=args.batch_size,size=args.size).create_reader()     
        epochs = args.epoch 
        
       # lowest_loss=eval_to_select_best_model(train_model,1000)[1]
        lowest_loss=10
       
        log = []
        
        with open('log.json','r') as f:
            data=json.load(f)
            log=data
            
        for i in range(args.epoch_num,epochs):
            start=time.time()
            acc_list=[]
            loss_list=[]
            info={'epoch_num':None,'iterations':[], 'train_avg_loss':None, 'val_avg_loss':None, 'train_acc':None, 'val_acc':None}
            for batch_id, data in enumerate(train_reader()):
                
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')
               
               # dy_x_data=random_noise(dy_x_data).astype('float32')
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                
                compute_start=time.time()
                out = train_model(img)
                compute_end=time.time()
                acc = fluid.layers.accuracy(input=out, label=label)
                acc_list.append(acc.numpy())
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                info['iterations'].append(avg_loss.numpy()[0].item())
                loss_list.append(avg_loss.numpy())
                avg_loss.backward()
                
               
                opt.minimize(avg_loss)
                train_model.clear_gradients()

                end=time.time()
                print("Loss at epoch {} step {}: loss:{}, acc: {},compute_time:{},total_time:{}"
                .format(i, batch_id, avg_loss.numpy(), acc.numpy(),compute_end-compute_start,end-start))    
                start=time.time()
                
            print('训练集正确率:{},训练集平均loss:{}'.format(np.mean(acc_list),np.mean(loss_list)))
            os.system('rm  ./model/latest.pdparams &&'+'ln -s /home/aistudio/model/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i)+'.pdparams '+args.save_dir + '/latest.pdparams')        
              
            fluid.dygraph.save_dygraph(train_model.state_dict(), args.save_dir + '/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i))
            
            _,loss,score= eval_to_select_best_model(train_model,lowest_loss)
            info['val_avg_loss']=loss.item()
            info['val_acc']=score.item()
            info['train_avg_loss']=np.mean(loss_list).item()
            info['train_acc']=np.mean(acc_list).item()
            info['epoch_num']=i
            log.append(info)
            
            if _:
             #   os.system('rm  ./model/best.pdparams &&'+'ln -s /home/aistudio/model/rep_model_'+args.dataset+'_'+str(args.size)+'_'+str(i)+'.pdparams '+args.save_dir + '/best.pdparams')        
                lowest_loss=loss
            else:
              #  model, _ = fluid.dygraph.load_dygraph(args.pretrain)
              #  train_model.load_dict(model)
                 pass
            train_model.train()
            with open('log.json','w') as f:
                json.dump(log,f)
            
            
def eval_to_select_best_model(model,pre_loss):
        
        model.eval()
     
        eval_reader =  reader.DS(args.test_file, 'data/data47656/hmdb/', model='2d', random=False,mode=args.mode, length=args.length, batch_size=args.batch_size,size=args.size).create_reader()
        acc_list = []
        loss_list=[]
        for batch_id, data in enumerate(eval_reader()):
            dy_x_data = np.array([x[0] for x in data]).astype('float32')
            y_data = np.array([[x[1]] for x in data]).astype('int64')
          #  dy_x_data=random_noise(dy_x_data).astype('float32')
            img = fluid.dygraph.to_variable(dy_x_data)
            label = fluid.dygraph.to_variable(y_data)
            out = model(img)
            acc = fluid.layers.accuracy(input=out, label=label)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)
            loss_list.append(avg_loss.numpy())
            acc_list.append(acc.numpy()) 
        score=np.mean(acc_list)
        loss=np.mean(loss_list)
        print("验证集准确率为:{},平均loss:{}".format(score,loss))
        if loss<pre_loss:
            return True,loss,score
        return False,loss,score           

if __name__ == "__main__":
    args = parse_args()
    
    if args.phase =='train':
       train(args)
    elif args.phase=='eval':
        eval(args)
