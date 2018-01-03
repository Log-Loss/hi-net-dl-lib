# Instruction
This project converts the parameters of layers and model to the **config** of the dl4j.

## Layers

#### Convolution Layer
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| filters    | number of filters applied to the layer | number [> 0] |  
| kernelSize | the kernel size  | number [eg. 3 for 3x3 >= 1 usually 1,3,5,7]|  
| strides | the strides applied  | number [eg. 2 for 2x2 >= 1] |  
| padding |  the padding applied  | number [>= 0] |
| activation | activation function of the layer  | string [relu, leakyrelu, tanh, sigmoid, none]| 

#### Dense Layer
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| outputDim  | output dimension of the layer | number [> 0] |  
| activation | activation function of the layer  | string [relu, leakyrelu, tanh, sigmoid]|  
| weightInit | the weight initialization method  |string [xavier, zero, uniform] |  

#### Pooling Layer
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| kernelSize | the kernel size  | number [eg. 3 for 3x3 >= 1 usually 1,3,5,7]|  
| strides | the strides applied  | number [eg. 2 for 2x2 >= 1] |  
| poolingType | the type of pooling | string [max, average]

#### Dropout Layer
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| dropoutRate | dropout rate | number [< 1]|  

#### Output Layer
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| outputNum | output dimension of the model | number [> 1]|
| activation | activation function of the layer | string [softmax, tanh, sigmoid] |
| lossFunction | loss function of the model | string [neg, sqrt]|


#### LSTM Layer

**the lstm layer is different from other layers and return config directly**  
    
**required parameters**   

| parameters | description | format |  
| ---------- | :-------: | ----: |  
| hiddenLayerWidth | the units in one hidden layer | number [>= 1] |
| hiddenLayerCount | the number of hidden layers | number [>= 1] |
| sequenceLen | the length of the input sequence | number [> 0] |
