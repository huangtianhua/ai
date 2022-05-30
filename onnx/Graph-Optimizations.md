# Graph Optimizations in ONNX Runtime

ONNX Runtime提供了各种图形优化来提高性能。图优化本质上是图级别的转换，包括小图简化、节点消除甚至是更复杂的节点融合和布局优化。

图形优化根据其复杂性和功能分为几个类别（或级别），有在线（online）和离线（offline）执行模式。在线优化需要在执行推理之前完成，离线模式下运行时会将优化后的图形保存到磁盘上。ONNX Runtime提供Python，C#，C++和C API，以启用不同的优化级别，并在离线模式与在线模式之间进行选择。

## Graph Optimization Levels

图形优化可以分为三种级别：

- Basic 基础级别
- Extended 扩展级别
- Layout Optimizations 布局优化

在应用当前级别的优化之前，会执行当前级别之前的优化（例如我们准备执行extended优化，Basic级别的优化会在执行extended优化之前先执行）。

默认情况下启用所有优化。

### Basic Graph Optimizations 基本图优化

保留语义的图重写，比如消除冗余节点和冗余计算，在图分区之前执行，适用于所以Execution providers，可用的基本图优化包括：
- Constant Folding 常量折叠：静态计算仅依赖常量初始值设定项的图形部分，进行优化后就可以不需要在运行时一次又一次的计算它们
- Redundant node eliminations 冗余节点消除：在不更改图形结构的情况下消除所有冗余节点，目前ONNX Runtime支持以下冗余节点的消除优化：
  - Identity Elimination
  - Slice Elimination
  - Unsqueeze Elimination
  - Dropout Elimination 

  
- Semantics-preserving node fusions 保留语义的节点融合，将多个节点融合/折叠为单个节点，例如，Conv Add融合会将Add运算符折叠为Conv运算符的偏差。当前ONNX Runtime支持以下融合优化：
  - Conv Add Fusion
  - Conv Mul Fusion
  - Conv BatchNorm Fusion
  - Relu Clip Fusion
  - Reshape Fusion

### Extended Graph Optimizations 扩展图优化

这类优化包括复杂的节点融合。它们在图分区之后运行，并且仅应用于分配给CPU或CUDA的execution providers的节点。可用的Extended图优化如下：

| Optimization | Execution Provider | Comment |
| ------------ | ------------------ | ------- |
|GEMM Activation Fusion| CPU |
|Matmul Add Fusion | CPU |
|Conv Activation Fusion | CPU |
|GELU Fusion | CPU or CUDA |
|Layer Normalization Fusion | CPU or CUDA |
|BERT Embedding Layer Fusion | CPU or CUDA | Fuse BERT embedding layer, layer normalization and attention mask length|
|Attention Fusion*|CPU or CUDA|
|Skip Layer Normalization Fusion|CPU or CUDA|Fuse bias of fully connected layer, skip connection and layer normalization|
|Bias GELU Fusion|CPU or CUDA|Fuse bias of fully connected layer and GELU activation|
|GELU Approximation*|CUDA| Disabled by default.Enable with kOrtSessionOptionsEnableGeluApproximation|

### Layout Optimizations 布局优化
此类优化更改了适用节点的数据layout，以实现更高的性能改进。 它们在图分区之后运行，并且仅适用于分配给CPU execution providers的节点。 可用的布局优化如下：
- NCHWc Optimizer: 使用NCHWc layout而不是NCHW layout
  
  - NHWC, 又称“channels_last”,是CPU指令比较适合的方式
  - NCHW，又称：“channels_first”,在GPU中，使用NCHW格式计算卷积速度更快
  - TensorFlow：缺省NHWC，GPU也支持NCHW
    Caffe：NCHW
    PyTorch：NCHW

  尽管存储的数据实际上是一样的，但是不同的顺序会导致数据的访问特性不一致，因此即使进行同样的运算，相应的计算性能也会不一样。对于"NCHW" 而言，其同一个通道的像素值连续排布，更适合那些需要对每个通道单独做运算的操作，比如"MaxPooling"。对于"NHWC"而言，其不同通道中的同一位置元素顺序存储，因此更适合那些需要对不同通道的同一像素做某种运算的操作，比如“Conv1x1”

  ![存储格式](../images/%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F.png)

  - NCHW是先取W方向数据；然后H方向；再C方向；最后N方向。所以，序列化出1D数据：
    000 (W方向) 001 002 003，(H方向) 004 005 … 019，(C方向) 020 … 318 319，(N方向) 320 321 …
  - NHWC是先取C方向数据；然后W方向；再H方向；最后N方向。所以，序列化出1D数据：
    000 (C方向) 020 … 300，(W方向) 001 021 … 303，(H方向) 004 … 319，(N方向) 320 340 …


#### 代码实现
以恒等冗余节点消除为例：/root/onnxruntime/onnxruntime/core/optimizer/identity_elimination.cc
```
/**
  Case to eliminate Identity node when 
  - the input nodearg has only one consumer, which is the Identity itself
  - the input def is not a graph output
  
  For examples: 

  OK to eliminate:
  
    Identity output is another node, and the Identity is the only consumer of X
      X ---> Identity ---> Y where Y could be graph output

    Identity input arg is not shared with other output arg of X
      + (arg0) ---> Identity0 ---> Z 
      |
      X (arg1) ---> Identity1 ---> Y

  Not OK to eliminate:

    Identity input arg, i.e., arg0, is also an input arg of other Identity
      + (arg0) ---> Identity0 ---> Z 
      |
      X (arg0) ---> Identity1 ---> Y

    Identity input def, i.e., def0, is also a graph output
      + (def0) ---> Z where Z is graph output
      |
      X (def0/arg0) ---> Identity ---> Y
 */
Status EliminateIdentity::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  if (!graph.NodeProducesGraphOutput(node)) {
    if (graph_utils::RemoveNode(graph, node)) {
      rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
    }
  } else {
    // keep a reference of output def to the graph output
    NodeArg* output = node.MutableOutputDefs()[0];
    const Node* p_input_node = graph_utils::GetInputNode(node, 0);
    // get mutable input node
    Node& input_node = *graph.GetNode(p_input_node->Index());
    int output_idx = graph_utils::GetNodeOutputIndexFromOutputName(input_node, node.MutableInputDefs()[0]->Name());
    // remove Identity node and its input edge
    graph.RemoveNode(node.Index());
    // update input node's output def to the graph output
    input_node.MutableOutputDefs()[output_idx] = output;
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }
  return Status::OK();
}
```

## Online/Offline Mode 在线、离线模式

所有优化均可在线或离线模式下执行。在线模式下，当初始化inference session时，我们会在执行模型推理之前应用所有启用的图优化。 当然这有个显著的问题就是每次启动inference session时都应用所有优化，会增加模型初始化时间的开销（尤其是对于复杂模型），这在生产环境中可能会有非常大的影响。但是离线模式下就可以带来避免初始化时间开销的优势，离线模式下，执行图形优化后，ONNX Runtime会将生成的优化后的模型保存到磁盘。随后，当为此模型创建新的inference session时，我们可以改用已经优化的模型来减少启动时间。

Notes:

  - 采用离线模式，确保和将运行模型推理的设备完全相同的配置项（例如，execution providers，optimization level优化等级）和硬件（例如，你不能在仅配备CPU的设备上运行针对GPU的execution providers预先优化的模型）。
  - 启用布局优化，离线模式只能在保存脱机模型时在与环境兼容的硬件上使用。 例如，如果模型具有针对AVX2的布局优化，则离线模型将需要支持AVX2的CPU。

## Usage 使用说明

### Level 优化级别
ONNX Runtime通过枚举`GraphOptimizationLevel`来确定将启用上述哪些优化级别，选择一个级别将启用该级别的优化，以及所有上一级别的优化。例如，启用扩展优化，也将启用基本优化。各优化级别到枚举的映射如下：
  - GraphOptimizationLevel::ORT_DISABLE_ALL -> 取消所有的优化
  - GraphOptimizationLevel::ORT_ENABLE_BASIC -> 使能基本优化
  - GraphOptimizationLevel::ORT_ENABLE_EXTENDED -> 使能基本和扩展优化
  - GraphOptimizationLevel::ORT_ENABLE_ALL -> 使能所有级别优化包括布局优化

### Online/Offline Mode 离线在线模式
要启用优化模型到磁盘的序列化，请将SessionOptions选项`optimized_model_filepath`设置为要存储优化模型的路径。

### PYTHON API EXAMPLE

```
import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "<model_output_path\optimized_model.onnx>"

session = rt.InferenceSession("<model_path>", sess_options)
```

### C++ API EXAMPLE
```
Ort::SessionOptions session_options;

// Set graph optimization level
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

// To enable model serialization after graph optimization set this
session_options.SetOptimizedModelFilePath("optimized_file_path");

auto session_ = Ort::Session(env, "model_file_path", session_options);
```