import numpy as np
import random
import sys
import os
#import lintel
import functools
import cv2
import paddle
from scipy import misc
from PIL import Image
dw=[]
class DS:

    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=True, c2i={},batch_size=16,buf_size=512,num_reader_threads=2,size=112):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.num_reader_threads=num_reader_threads
        self.model = model
        self.size = size
       
        self.buf_size=buf_size
        self.batch_size=batch_size
        self.filelist=split_file
        self.data=[]
        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v,c = l.strip().split(' ')
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                self.data.append([os.path.join(root, v), self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random

    


   

    def create_reader(self):
        _reader = self._reader_creator(self.filelist, 
                                       shuffle=self.random,
                                       num_threads=self.num_reader_threads,
                                       buf_size=self.buf_size)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader

    def _reader_creator(self,
                        split_file,
                        shuffle=False,
                        num_threads=1,
                        buf_size=1024
                        ):
        def reader():
            with open(split_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    vid = line.strip()
                    yield vid
        def load_frames_from_video(vid_file,length):
            cap=cv2.VideoCapture(vid_file)
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            t=total-length*2
           
            import random
            if t>0:
               start=random.randint(0,t)
               step=2
            elif total>length:
                start=random.randint(0,total-length)
                step=1
            else:
                start=0
                step=1
            if not self.random:
                start=t//2 if t>=0 else 0
                start=(total-length)//2 if t<0 and total>length else 0
                step=2 if t>=0 else 1
            list_img=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,start)
            cap.grab()
            for i in range(length*step if length< total else total):
               
                success,frame=cap.read()
                if success and (start+i)%step==0:
                     list_img.append(np.array(frame))
                if not success:
                    break        
            cap.release()
            res=length-len(list_img)
            for i in range(res):
                
                list_img.append(list_img[-1])
            
            return np.array(list_img)

        def resize(group_img,size):
                
                n,h,w,c=group_img.shape
                if 224>h or 224>w:
                    newgroup=[]
                    for i in range(n):
                        new_img = Image.fromarray(np.uint8(group_img[i])).resize((size,size))
                        new_img=np.array(new_img)
                        newgroup.append(new_img)
                    
                    newgroup=np.array(newgroup)
                    return newgroup
               
                if not self.random:
                  if self.size==112:
                      w=w//2
                      h=h//2
                      i = int(round((h-self.size)/2.))
                      j = int(round((w-self.size)/2.))
                      df = np.reshape(group_img, newshape=(n, h*2, w*2, 3))[:,::2,::2,:][:, i:-i, j:-j, :]
                  else:
                      
                      i = int(round((h-self.size)/2.))
                      j = int(round((w-self.size)/2.))
                      df = np.reshape(group_img, newshape=(n, h, w, 3))[:, i:-i, j:-j, :]
                      
                else:
                  if self.size==112:
                      w=w//2
                      h=h//2
                      th = self.size
                      tw = self.size
                      i = random.randint(0, h - th) if h!=th else 0
                      j = random.randint(0, w - tw) if w!=tw else 0
                      df = np.reshape(group_img, newshape=(n, h*2, w*2, 3))[:,::2,::2,:][:, i:i+th, j:j+tw, :]
                  else:
                      
                      th = self.size
                      tw = self.size
                      i = random.randint(0, h - th) if h!=th else 0
                      j = random.randint(0, w - tw) if w!=tw else 0
                      df = np.reshape(group_img, newshape=(n, h, w, 3))[:, i:i+th, j:j+tw, :]
                return df
        def group_random_flip(img_group):
            v = random.random()
            if v < 0.5:
                ret = [np.array(Image.fromarray(np.uint8(img)).transpose(Image.FLIP_LEFT_RIGHT) )for img in img_group]
                return np.array(ret)
            else:
                return img_group
       
        def video_reader(vid):
             
              vid, cls = vid.split(' ')      
                       
              df=load_frames_from_video(self.root+vid,self.length)    
              df=resize(df,self.size) 
              df=group_random_flip(df) if self.random else df
              if self.mode == 'flow':
                  
                  # only take the 2 channels corresponding to flow (x,y)
                  df = df[:,:,:,1:]
                  if self.model == '2d':
                      # this should be redone...
                      # stack 10 along channel axis
                      df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                      df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
                  
                      
              df = 1-2*(df.astype(np.float32)/255)      

              if self.model == '2d':
                  # 2d -> return TxCxHxW
                  return df.transpose([0,3,1,2]), cls
              # 3d -> return CxTxHxW
              return df.transpose([3,0,1,2]), cls
              
        mapper = functools.partial(
            video_reader )

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)

        def __len__(self):
          return len(self.data)

if __name__ == '__main__':
  with paddle.fluid.dygraph.guard(paddle.fluid.CPUPlace()):
  
   
    dataseta = DS('/home/aistudio/train.list', 'data/data48916/', random=True,model='3d', mode='flow',size=224, length=16, batch_size=32).create_reader()
    dataseta1 = DS('/home/aistudio/new_train.txt', 'data/data47656/hmdb/', random=True,model='2d', mode='rgb',size=224, length=24, batch_size=2).create_reader()
 #   dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)
    for j in range(1):
        for i,data in enumerate(dataseta1()):
            x = np.array([x[0] for x in data]).astype('float32')
            y = np.array([[x[1]] for x in data])
           # print(i,x.shape)
            del x,y
