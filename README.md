# caffe2keras
This code is for converting a caffe model to keras model and load caffe weights   

## Layer Supported
- Dense
- Maxpooling2D, AveragePooling2D
- Conv2D
- Batch_Normaliztion
- Softmax, ReLu
- Dropout
- Add

## To Do
- Add more layers
- More effiency code

## Warning
- If you're going to convert a model with batch_norm and scale layers, make sure their layer name in this style:  
  - 'bn' + (something same)  
  - 'scale' + (something same)
  - example: bnbranch2\_conv1, scalebranch2\_conv1  
- If you get problems in loading caffe model like me, you can just delete data, loss, accuracy type layers, it doesn't useful after converting to keras model.
- If you don't want to load the weights in scale layers, ignore the first warning

## Instruction 
``` 
  python caffe2keras.py
  --model "caffe_model_path"  #  this must be specified  
  --weights "caffe_weights_path"  # this is optional, if none, you will get a raw model without pretrained weights  
  --output "save_path_for_keras_model" # this must be specified  
  --log "True or False" # default:False, if True, will show some info about build model and load weights.   
  It's a good choice if   it's your first time to use it or get some wrong with the saved keras model  
```
