# 概述
ONNX runtime是一个跨平台的机器学习模型加速器（cross-platform machine-learning model accelerator），其接口灵活可以集成硬件相关库（hardware-specific libraries）。
refer to: https://github.com/microsoft/onnxruntime

![ONNX Runtime与训练框架以及运行目标机器/环境的关系](https://www.onnxruntime.ai/images/ONNX_Runtime_EP1.png)
可以看出ONNX Runtime是在PyTorch、TensorFlow等训练框架导出ONNX模型后作为ONNX Runtime的输入，再在指定的硬件平台去运行模型，底层适配不同的运行环境（OS、硬件（CPU、GPU、FPGA、NPU））并做底层优化，使不同平台训练出的模型可以再同一个运行时中运行。

![ONNX runtime的输入、输出以及内部结构](https://www.onnxruntime.ai/images/ONNX_Runtime_EP3.png)
ONNX模型以及输入数据作为ONNX runtime的输入，内部对图形做运算，并根据运行环境调用不同的内核运行相关的指令，即图中描述的不同的Execution Provider。


**ONNX Runtime inference推理**可以带来更好的用户体验以及更低的消耗，支持深度学习框架（deep learning frameworks）例如PyTorch、TensorFlow/Keras以及一些经典的的机器学习库例如scikit-learn,LightGBM,XGBoost等。ONNX runtime兼容不同的硬件、驱动和操作系统，沿着图形优化和转换利用硬件加速的路线来提供更好的性能。

**ONNX Runtime training训练**仅需要在现有的PyTorch训练脚本上额外增加一行就可以在多节点NVIDIA GPU上对转换模型（transformer model）加速模型训练的时间。

# Get started with ORT for Python
refer to:https://onnxruntime.ai/docs/get-started/with-python.html

## 安装ORT
两个安装包、GPU版本和CPU版本，二选一
- GPU版本：pip install onnxruntime-gpu
- CPU版本（运行Arm CPU或者macOS）：pip install onnxruntime

## 安装用于模型导出的ONNX
```python
#安装PyTorch
pip install torch
#安装PyTorch视觉模型库
pip install torchvision
```
## 基于PyTorch的样例
### PyTorch CV（Computer Vision，机器视觉）
```python
import os.path as osp
import onnx
import torch
import torchvision
import onnxruntime as ort
import numpy as np

# test input array
test_arr = np.random.randn(10, 3, 224, 224).astype(np.float32)

dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
model =  torchvision.models.resnet50(pretrained=False).cpu().eval()
print('pytorch result:', model(torch.from_numpy(test_arr).cpu()))

input_names = ['input']
output_names = ['output']

if not osp.exists('resnet50.onnx'):
    # translate pytorch model into onnx
    torch.onnx.export(model, dummy_input, "resnet50.onnx", verbose=True, input_names = input_names, output_names = output_names)

model = onnx.load('resnet50.onnx')
onnx.checker.check_model(model)

ort_session = ort.InferenceSession('resnet50.onnx')
outputs = ort_session.run(None, {'input' : test_arr})

print('onnx result', outputs[0])
```

*代码走读备注：*
*torch.randn：返回一个由均值为0、方差为1的正态分布（也称为标准正态分布）中的随机数填充的张量；入参size(int...)为定义输出张量的shape数列；参数device为期望返回张量的设备*
*model =  torchvision.models.resnet50(pretrained=False).cpu().eval()函数调用中：cpu()方法代表将模型所有的参数和Buffer移到CPU上；eval()方法代表将模块设置为评估模式（evaluation mode），只有特定的模块才有效，等同于调用self.train(False) <torch.nn.Module.train>*



#### 主要操作步骤：
1.使用torch.onnx.export导出模型(Export the model using torch.onnx.export)
参数说明： 
```python
torch.onnx.export(model,            # 要导出的pytorch模型对象
                  dummy_input,      # 模型的输入（或者多输入元组a tuple for multiple inputs）
                  "resnet50.onnx",  # 模型导出的存储路径+文件名
                  verbose = True,   # 如果为True，代表打印待被导出的模型描述到stdout，默认为Flase
                  input_names = input_names,      #模型的输入参数
                  output_names = output_names     #模型的输出参数
)
```

2、使用onnx.load加载并验证onnx模型(Load the onnx model with onnx.load)
```python
model = onnx.load('resnet50.onnx')
onnx.checker.check_model(model)
```

3、使用ort.InferenceSession创建推理会话
ort_session = ort.InferenceSession('resnet50.onnx')
outputs = ort_session.run(None, {'input' : test_arr})
print('onnx result', outputs[0])



# ONNX runtime API
refer to: https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs
## API概览
ONNX Runtime用ONNX图形的格式或者ORT格式（内存和硬盘的受限环境内）加载和运行推理模型。模型数据可用最佳的方式指定或者访问，用以匹配对应的场景。
#### 加载和运行模型
 InferenceSession是ONNX runtime的主类，用于加载和运行ONNX模型，同时可指定环境和应用配置参数。
```python
session = onnxruntime.InferenceSession('model.onnx')
outputs = session.run([output names], inputs)
```
 ONNX和ORT格式模型由计算图形组成，用操作符建模，对不同的硬件运行环境用对应优化的操作符内核（optimized operator kernel）来实现。ONNX Runtime通过execution provider来协调operator kernel的执行。一个execution provider包含一个目标执行环境（如CPU、GPU、IoT等）上的一个kernel集合。Execution Provider使用providers参数来配置。Execution provider的优先级取决于providers参数列表的顺序。
如下示例如果有CUDA execution provider这个kernel，则ONNX runtime在GPU上执行，如果没有则在CPU上执行。
```python
session = onnxruntime.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```
providers参数如果不指定，则默认会使用运行环境上所有可用的provider，并使用默认的优先级。

可以用sesion options参数指定会话参数，例如打开会话的跟踪profiling，示例代码如下。
```python
options = onnxruntime.SessionOptions()
options.enable_profiling=True
session = onnxruntime.InferenceSession('model.onnx', 
             sess_options=options, 
             providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
```

### 数据输入和输出
ONNX Runtimes Inference Session使用OrtValue类来做输入输出。
#### CPU上的数据
CPU环境下（默认），OrtValues可以映射为本地Python数据结构：numpy arrays, numpy arrays类型的dictionaries和lists。
```python
# X is numpy array on cpu
ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X)
ortvalue.device_name()  # 'cpu'
ortvalue.shape()        # shape of the numpy array X
ortvalue.data_type()    # 'tensor(float)'
ortvalue.is_tensor()    # 'True'
np.array_equal(ortvalue.numpy(), X)  # 'True'

# ortvalue can be provided as part of the input feed to a model
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
results = session.run(["Y"], {"X": ortvalue})
```
默认的，ONNX Runtime总是在CPU上存储输入、输出。如果输入、输出是在一个device上时，CPU和设备之间会有数据备份，CPU存储在CPU上并非好的选择。

#### 设备上的数据
ONNX运行时支持自定义数据结构，该结构支持所有ONNX数据格式，允许用户将支持这些格式的数据放在设备上，例如在支持CUDA(CUDA（Compute Unified Device Architecture），统一计算设备框架，是显卡厂商NVIDIA推出的运算平台)的设备上。在ONNX Runtime里称为IOBinding。

使用IOBinding特性时，需要改用InferenceSession.run_with_iobinding()接口。

示例1，图形运行在CUDA上， 用户可以使用IOBinding拷贝数据到GPU上。
```python
# X is numpy array on cpu
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
io_binding = session.io_binding()
# OnnxRuntime will copy the data over to the CUDA device if 'input' is consumed by nodes on the CUDA device
io_binding.bind_cpu_input('input', X)
io_binding.bind_output('output')
session.run_with_iobinding(io_binding)
Y = io_binding.copy_outputs_to_cpu()[0]
```
*接口解读：
1.bind_cpu_input方法用于绑定一个输入到CPU上的数组。其中'input'为模型的输入参数，X为CPU上python数组输入数据值
2.bind_output方法用于绑定输出参数。其中'output'为模型的输出参数，不指定输出的device则默认为cpu设备*

示例2，输入数据在device上，用户可以直接使用输入。输出数据在CPU上。
```python
# X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
io_binding = session.io_binding()
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_output('output')
session.run_with_iobinding(io_binding)
Y = io_binding.copy_outputs_to_cpu()[0]
```
*接口解读：bind_input中直接指定绑定模型的input到X_ortvalue变量对应的设备上,而X_ortvalue是在CUDA设备上分配的*

示例3，输入输出都在device上，用户可以直接在device上使用输入输出。
```python
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
io_binding = session.io_binding()
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_output(name='output', device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())
session.run_with_iobinding(io_binding)
```

示例4，用户可以请求ONNX Runtime在设备上分配输出，对动态shape输出非常有用。用户可以使用get_outputs() API访问分配的输出的OrtValues对象。
```python
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
io_binding = session.io_binding()
io_binding.bind_input(name='input', device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
#Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
io_binding.bind_output('output', 'cuda')
session.run_with_iobinding(io_binding)
# The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
ort_output = io_binding.get_outputs()[0]
```
示例5，ONNX Runtime支持在模型推理时直接使用OrtValues作为输入的一部分。
```python
#X is numpy array on cpu
#X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, 'cuda', 0)
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)  # Change the shape to the actual shape of the output being bound
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
io_binding = session.io_binding()
io_binding.bind_ortvalue_input('input', X_ortvalue)
io_binding.bind_ortvalue_output('output', Y_ortvalue)
session.run_with_iobinding(io_binding)
```

示例6，也可以直接将输入输出绑定为PyTorch的tensor
```python
# X is a PyTorch tensor on device
session = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
binding = session.io_binding()

X_tensor = X.contiguous()

binding.bind_input(
    name='X',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(x_tensor.shape),
    buffer_ptr=x_tensor.data_ptr(),
    )

## Allocate the PyTorch tensor for the model output
Y_shape = ... # You need to specify the output PyTorch tensor shape
Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').contiguous()
binding.bind_output(
    name='Y',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(Y_tensor.shape),
    buffer_ptr=Y_tensor.data_ptr(),
)

session.run_with_iobinding(binding)
```
