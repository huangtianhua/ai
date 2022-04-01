# Ascend资料

## 内容

[昇腾产品工作模式](./ascend_work_model.md)

## 链接

- [昇腾社区](https://www.hiascend.com/)

## 名词

**FLOPS**：Floating-point operations per second, 每秒执行的浮点运算次数。根据速度在头部加单位（M、G、T、P、E）分别表示百万MFLOPS（10^6）、十亿GFLOPS（10^9）、万亿TFLOPS（10^12）、千万亿PFLOPS（10^15）、百亿亿EFLOPS（10^18）

**FP16/32/64**：浮点编码类型，FP16表示以2字节（16位）进行编码存储，同理FP32表示以4字节（32位）进行编码存储。

**ACL**：Ascend Computing Language，昇腾芯片计算加速库（CANN）提供的开发库之一，提供设备管理、运行资源上下文管理、运行流管理、内存管理、模型/算子的加载和执行、数据处理的功能。

ACL几个概念对象：

    - Host: 连接AI芯片的主机
    - Device: AI芯片设备，如昇腾310
    - Context: 作为一个容器，管理上下文对象的生命周期，不同的Conetxt的Stream、Event是完全隔离的
    - Stream: 用于维护一些异步操作的执行顺序，确保代码按照应用中的调用顺序在Device上执行
    - Event: 支持调用ACL接口同步Stream之间的任务，包括同步Host与Device之间的任务、Device与Device间的任务。例如，若Stream2的任务依赖Stream1的任务，可以创建一个Event，将它插入到Stream1，在执行Stream2之前，先同步等待Event完成
    - AIPP: AI processing，用于完成图像预处理。

ACL函数命名规则:

    acl<接口类型><接口功能>
    接口类型：
      - rt: Runtime 运行管理类
      - dvpp: DVPP(Digital Vision Preprocessing) 媒体数据处理类
      - aipp: AIPP AI Preprocessing类
      - blas: CBLAS BLAS类
      - mdl: Model 模型推理类
      - grph: Graph 图类
      - drv: Driver 驱动类
      - op: OP 算子类
    例如:
      aclrtGetDeviceCount(uint32_t *count)
      aclrtGetRunMode(aclrtRunMode *runMode)
      aclrtSetDevice(int32_t deviceld)
      aclrtMallocHost(void **hostPtr, size_t size)
      aclmdlLoadFromFile(const char *modelPath, uint32_t *modleld)

**TBE**: Tensor Boost Engine, 华为自研的从TVM扩展来的基于昇腾NPU的算子开发工具

**MindX**: 华为自研的基于CANN的应用开发套件，对CANN的ACL进行了封装，方便用户快速开发AI应用。

**Ascend**: 昇腾芯片是华为自研的基于达芬奇架构的NPU，主要型号有：

  - Ascend 310： 推理芯片
  - Ascend 710： 训练、推理芯片
  - Ascend 910： 昇腾系列最强训练、推理芯片
  - Ascend 320 610 920: 下一代昇腾芯片，计划中

**Atlas**: 基于昇腾芯片的的AI基础设施解决方案，主要产品有：

  - Atlas 200：AI智能模块，集成了昇腾310芯片
  - Atlas 200 DK：AI应用开发板，集成了昇腾310芯片
  - Atlas 300：基于昇腾芯片的PCI加速卡，拥有多个型号：300I(昇腾310)、300I Pro（昇腾710）、300V Pro（昇腾710，面向视频场景）、300T（昇腾910）
  - Atlas 500：面向边缘应用的服务器产品，包括智能小站（昇腾310）和智能边缘服务器（鲲鹏920+昇腾300I）
  - Atlas 800：有两款，1是基于昇腾310处理器的推理服务器，可插Atlas 300I推理卡；2是基于昇腾910+鲲鹏920的训练服务器
  - Atlas 900 PoD：基于64个昇腾910 + 32个鲲鹏920处理器的AI训练集群基础单元
  - Atlas 900 AI 集群：由数千颗昇腾训练处理器构成的AI集群，相当于 50 万台高性能 PC 的计算能力

## 下载链接

- [CANN社区版](https://www.hiascend.com/software/cann/community)(商用版为收费软件)

  一共4个软件包，分别是：
  - nnae: Neural Network Acceleration Engine 深度学习引擎包，支持离线、在线推理、训练。
  - nnrt: Neural Network Runtime 离线推理引擎包，仅支持离线推理
  - toolkit: 昇腾开发实用工具工具包
  - amct: Ascend Model Compression Toolkit 模型压缩工具，支持压缩pytorch、onnx、tensorflow等模型

- [CANN官方文档](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373)
- [CANN示例代码](https://gitee.com/ascend/samples)
