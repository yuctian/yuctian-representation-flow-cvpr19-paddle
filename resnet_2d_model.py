from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from paddle.fluid.dygraph import BatchNorm,Conv2D,Dropout,Linear
from paddle import fluid
from paddle.fluid import layers
import paddle
import numpy as np
import math

import rep_flow_layer as rf


import numpy as np

import os
import sys

class Bottleneck(fluid.dygraph.Layer):
  expansion = 4
  def __init__(self, inputs, filters, is_training, strides,
               use_projection=False, T=3, data_format='channels_last', non_local=False):
    
    super(Bottleneck, self).__init__()
    self.use_projection=use_projection
    filters_out = 4 * filters
    self.conv2d_1 =None if not  (strides != 1 or inputs != filters_out) else fluid.dygraph.Sequential(
                Conv2D(inputs, filters_out,filter_size=1, stride=strides, padding=0, bias_attr=fluid.ParamAttr(trainable=False)),
                BatchNorm(filters_out))
    self.conv2d_2=Conv2D(
                             inputs, filters, filter_size=1, stride=1,
                             param_attr=fluid.initializer.MSRAInitializer(uniform=False),
                             bias_attr=fluid.ParamAttr(trainable=False))
    
    self.bn_2=BatchNorm(filters,act='relu')
    
    self.conv2d_3=Conv2D(filters, filters, filter_size=3, stride=(strides,strides), padding=1,
    param_attr=fluid.initializer.MSRAInitializer(uniform=False),
    bias_attr=fluid.ParamAttr(trainable=False))
    self.bn_3=BatchNorm(filters,act='relu')
    self.conv2d_4=Conv2D(
                             filters, filters*4, filter_size=1, stride=1, padding=0,
                             param_attr=fluid.initializer.MSRAInitializer(uniform=False),
                             bias_attr=fluid.ParamAttr(trainable=False))
    self.bn_4=BatchNorm(4*filters)
         

  def forward(self, x):
    
    if  self.conv2d_1:
      res = self.conv2d_1(x)
      

    else:
      res = x
    y=self.conv2d_2(x)
    y=self.bn_2(y)
    y=self.conv2d_3(y)
    y=self.bn_3(y)
    y=self.conv2d_4(y)
    y=self.bn_4(y)
    return layers.relu(res+y)


  
class Block(fluid.dygraph.Layer):
  def __init__(self, inputs, filters, block_fn, blocks,block_ind, strides, is_training, name,
                   data_format='channels_last', non_local=0):
    super(Block, self).__init__()

    self.blocks = []


    # Only the first block per block_group uses projection shortcut and strides.
    self.blocks.append(self.add_sublayer(
                                   'bb_%d_%d' % (block_ind, 0),block_fn(inputs, filters, is_training, strides,
                                    use_projection=True, data_format=data_format)))
    inputs = filters * block_fn.expansion
    T = 3
    for i in range(1, blocks):
     
      self.blocks.append(self.add_sublayer(
                                   'bb_%d_%d' % (block_ind, i),
                                    block_fn(inputs, filters, is_training,1, T=T,
                                     data_format=data_format)))
      # only use 1 3D conv per 2 residual blocks (per Non-local NN paper)
      T = (3 if T==1 else 1)
    

  def forward(self, x):
   
    for block in self.blocks:
    
       x = block(x)
    return x

class ResNet(fluid.dygraph.Layer):
  
  def __init__(self, block_fn, layers, num_classes,
               data_format='channels_last', non_local=[], rep_flow=[],size=112,
               dropout_keep_prob=0.5,batch_size=16):
    """Generator for ResNet v1 models.
  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
    """
    super(ResNet, self).__init__()
    is_training = False 
    

    self.stem = Conv2D(3, 64, filter_size=7, bias_attr=fluid.ParamAttr(trainable=False),stride=2,padding=3,
    param_attr=fluid.initializer.MSRAInitializer(uniform=False))
    
    self.bn1 = BatchNorm(64, act='relu')
    self.num_classes=num_classes

    self.rep_flow = rf.FlowLayer(block_fn.expansion*128,bn=False,batch_size=batch_size)
    self.rep_flow2 =rf.FlowLayer(512,batch_size=batch_size)
    # res 2
    inputs = 64
    self.res2 = Block(block_ind=1,
      inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
      strides=1, is_training=is_training, name='block_group1',
      data_format=data_format, non_local=non_local[0])
 
    # res 3
    inputs = 64*block_fn.expansion
    self.res3 = Block(block_ind=2,
      inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
      strides=2, is_training=is_training, name='block_group2',
      data_format=data_format, non_local=non_local[1])

    # res 4
    inputs = 128*block_fn.expansion
    self.res4 = Block(block_ind=3,
      inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
      strides=2, is_training=is_training, name='block_group3',
      data_format=data_format, non_local=non_local[2])

    # res 5
    inputs = 256*block_fn.expansion
    self.res5 = Block(block_ind=4,
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_group4',
        data_format=data_format, non_local=non_local[3])

    self.dropout = Dropout(0.9)

    self.classify =  Linear(input_dim=512*block_fn.expansion,
                          output_dim=num_classes,
                          act='softmax',
                         param_attr=fluid.ParamAttr(
                          initializer=fluid.initializer.MSRAInitializer(uniform=True)))               

  def forward(self, x):
    x=layers.transpose(x,perm=[0,2,1,3,4])
    x= fluid.layers.pool3d(x,pool_size=(3,1,1),pool_type='avg',pool_stride=(2,1,1))
    b,c,t,h,w = x.shape
    x=layers.transpose(x,perm=[0,2,1,3,4])
    x=layers.reshape(x,shape=[b*t,c,h,w])
    x = self.stem(x)
   # print(self.stem.weight.numpy().sum())
    x = self.bn1(x)
    x =  layers.pool2d(x, pool_size=3, pool_type='max',pool_stride=2, pool_padding=1)
    x = self.res2(x)
    x = self.res3(x)
    bt,c,h,w = x.shape
    x= layers.reshape(x,shape=[b,t,c,h,w])
    x=layers.transpose(x,perm=[0,2,1,3,4])
    x= fluid.layers.pool3d(x,pool_size=(3,1,1),pool_type='avg',pool_stride=(2,1,1))
    b,c,t,h,w = x.shape
    x=layers.transpose(x,perm=[0,2,1,3,4])
    res=layers.reshape(x[:,1:-1],shape=[-1,c,h,w])
    x=layers.reshape(x,shape=[b*t,c,h,w])
    x = self.rep_flow(x)
    x = self.rep_flow2(x)
    x = layers.relu(res+x)
    x = self.res4(x)
    x = self.res5(x)
    
    x = self.dropout(x)
    x=layers.reduce_mean(x,dim=3)
    x=layers.reduce_mean(x,dim=2)
    
    x=layers.reshape(x,shape=[x.shape[0],-1])
    x = self.classify(x)
    
    x = layers.reshape(x,shape=[b,-1,self.num_classes])
    
    x =layers.reduce_mean(x,dim=1)
    return x


def resnet_2d_v1(resnet_depth, num_classes, data_format='channels_last', size=224,non_local=[0,0,0,0], rep_flow=[0,0,0,0,0],batch_size=16):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
      101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
      152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
      200: {'block': Bottleneck, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return ResNet(
    params['block'], params['layers'], num_classes, data_format, non_local, rep_flow,size=size,batch_size=batch_size)
