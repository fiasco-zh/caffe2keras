from keras.activations import relu, softmax
from keras.models import Input, Model
from keras.layers import Dense, BatchNormalization, Conv2D, Activation, Dropout, Add, MaxPooling2D, AveragePooling2D, Flatten
from keras import backend as K
import numpy as np

def get_bottom(bottom_name, layers):
    try:
        for key in layers.keys():
            if bottom_name in key:
                bottom = layers[key]
    except KeyError:
        print('KEY ERROR:please load bottom layer: {} before this layer'.format(bottom_name))  
    return bottom

def get_layer(layer_params, layers, verbose=False):
    layer_type = layer_params['type'].lower()
    layer_name = layer_params['name']  
    if layer_type == 'input':
        input_dim = np.array(layer_params['input_dim'][1:])
        if K.backend()=='tensorflow':
            input_dim = input_dim[[1, 2, 0]]
        layer = Input(tuple(input_dim), name='data')
        layers[layer.name] = layer
        return layer 
    
#Convolution layer
    elif layer_type == 'convolution':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
        kernel_size = layer_params['kernel_size']
        num_output = layer_params['num_output']
        pad = layer_params['pad']
        stride = layer_params.get('stride', 1)
        use_bias = layer_params.get('bias_term', True)
        padding = 'same' if pad==(kernel_size-1)/2 else 'valid'
        layer = Conv2D(num_output, kernel_size, strides=(stride, stride), padding=padding, name=layer_name,
                       use_bias=use_bias)(bottom)
        layers[layer.name] = layer
        
        return layer
    
#Dense layer
    elif layer_type == 'innerproduct':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
        num_output = layer_params['num_output']
        bottom_dim = bottom.shape.as_list()
        if len(bottom_dim)!=2:
            bottom = Flatten()(bottom)
        layer = Dense(num_output, name=layer_name)(bottom)
        layers[layer.name] = layer
        return layer
    
#Batch_norm layer
    elif layer_type == 'batchnorm':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
#         num_output = layer_params['num_output']
        layer = BatchNormalization(name=layer_name)(bottom)
        layers[bottom_name] = layer
        return layer
    
#Pool layer
    elif layer_type == 'pooling':
        bottom_name = layer_params['bottom'][0]
        pool = layer_params['pool']
        stride = layer_params.get('stride', 2)
        kernel_size = layer_params['kernel_size']
        bottom = get_bottom(bottom_name, layers)
        if pool=='MAX':
            layer = MaxPooling2D(pool_size=(kernel_size,kernel_size), strides=(stride,stride),
                                 name=layer_name)(bottom)
        elif pool=='AVE':
            layer = AveragePooling2D(pool_size=(kernel_size,kernel_size), strides=(stride,stride),
                                     name=layer_name)(bottom)
        layers[layer.name] = layer
        return layer
    
#Softmax layer
    elif layer_type == 'softmax':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
        layer = Activation(softmax, name=layer_name)(bottom)
#         layers[layer.name] = layer
        layers[bottom_name] = layer
        return layer
    
    elif layer_type == 'relu':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
        layer = Activation(relu, name=layer_name)(bottom)
#         layers[layer.name] = layer
        layers[bottom_name] = layer
        return layer

#Dropout layer
    elif layer_type == 'dropout':
        bottom_name = layer_params['bottom'][0]
        bottom = get_bottom(bottom_name, layers)
        dropout = 1 - float(layer_params['dropout_ratio'])
        layer = Dropout(dropout, name=layer_name)(bottom)
#         layers[layer.name] = layer
        layers[bottom_name] = layer
        return layer
    
#Add layer
    elif layer_type == 'eltwise':
        bottom_name1 = layer_params['bottom'][0]
        bottom_name2 = layer_params['bottom'][1]
        bottom1 = get_bottom(bottom_name1, layers)
        bottom2 = get_bottom(bottom_name2, layers)
        
        layer = Add(name=layer_name)([bottom1, bottom2])
        layers[layer.name] = layer
        return layer
    
    else:
        if verbose:
            print("skipped {} type layer, please check the model later, may be it's not right builded".format(layer_type))