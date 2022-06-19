# Development

## Model

### PyTorch Code

```python
import torch

class Debug(torch.nn.Module):
    def __init__(self):
        super(Debug, self).__init__()
        self.p1 = torch.randint(low=5, high=6, size=(3,), dtype=int)
        self.p2 = torch.randint(low=3, high=4, size=(3,), dtype=int)
        self.p3 = torch.randint(low=3, high=4, size=(3,), dtype=int)

    def forward(self, x):
        x = torch.add(x, self.p1)
        x = torch.add(x, self.p2)
        x = torch.add(x, self.p3)
        return x


net = Debug()
model_name = 'debug.onnx'
dummy_input= torch.randint(low=1, high=3, size=(3,), dtype=int)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'], opset_version=13)
```

### ONNX Model Structure

![debug01 onnx](https://user-images.githubusercontent.com/85333600/174466504-7f829c6a-cd13-45fe-ae83-eb4d660a52df.png)

**imports: ai.onnx v13**

**type: int64**

## Operator

### ONNX

| Operator  | Op Version  | Supported Type                                                  |
| :---      | :---        | :---                                                            |
| Add       | 1,6,7,13,14 | uint32, uint64, int32, int64, float16, float, double, bffloat16 |
| Sub       | 1,6,7,13,14 | uint32, uint64, int32, int64, float16, float, double, bffloat16 |
| Mul       | 1,6,7,13,14 | uint32, uint64, int32, int64, float16, float, double, bffloat16 |

### CANN Support

| Operator  | Supported Version | Supported Type                                      |
| :---      | :---              | :---                                                |
| Add       | 8,9,10,11,12,13   | int8,int16,int32,int64,uint8,float32,float16,double |
| Sub       | 8,9,10,11,12,13   | int32,float16,float32                               |
| Mul       | 8,9,10,11,12,13   | float16,float32,uint8,int8,int16,int32              |

## ONNXRUNTIME

![Graph drawio](https://user-images.githubusercontent.com/85333600/174468439-e9f3fca9-b30b-4632-a3e8-01fdf89cb8a6.png)

## CANN

```c
aclError aclopCompileAndExecute(const char *opType,
                                int numInputs,
                                const aclTensorDesc *const inputDesc[],
                                const aclDataBuffer *const inputs[],
                                int numOutputs,
                                const aclTensorDesc *const outputDesc[],
                                aclDataBuffer *const outputs[],
                                const aclopAttr *attr,
                                aclopEngineType engineType,
                                aclopCompileType compileFlag,
                                const char *opPath,
                                aclrtStream stream);
```

| 参数名       | 输入/输出  | 说明                                                            |
| :---         | :---      | :---                                                            |
| opType       | 输入      | 算子类型名称的指针 |
| numInputs    | 输入      | 算子输入tensor的数量 |
| inputDesc    | 输入      | 算子输入tensor描述的指针数组。需提前调用aclCreateTensorDesc接口创建aclTensorDesc类型。inputDesc数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应 |
| inputs       | 输入      | 算子输入tensor的指针数组。需提前调用aclCreateDataBuffer接口创建aclDataBuffer类型的数据。inputs数组中的元素个数必须与numInputs参数值保持一致，且inputs数组与inputDesc数组中的元素必须一一对应 |
| numOutputs   | 输入      | 算子输出tensor的数量 |
| outputDesc   | 输入      | 算子输出tensor描述的指针数组。需提前调用aclCreateTensorDesc接口创建aclTensorDesc类型。outputDesc数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应 |
| outputs      | 输入&输出 | 算子输出tensor的指针数组。需提前调用aclCreateDataBuffer接口创建aclDataBuffer类型的数据。outputs数组中的元素个数必须与numOutputs参数值保持一致，且outputs数组与outputDesc数组中的元素必须一一对应 |
| attr         | 输入      | 算子属性的指针。需提前调用aclopCreateAttr接口创建aclopAttr类型 |
| engineType   | 输入      | 算子执行引擎 |
| compileFlag  | 输入      | 算子编译标识 |
| opPath       | 输入      | 算子实现文件*.py路径的指针，不包含文件名。如果将compileFlag设置为ACL_COMPILE_UNREGISTERED 时，需要设置opPath |
| stream       | 输入      | 该算子需要加载的stream |

## Implement

binary_elementwise_ops.h
```c++
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cann/cann_kernel.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cann {

struct BinaryElementWiseCannPreparation {
  BinaryElementWiseCannPreparation () {
    opAttr_ = aclopCreateAttr();
  }

  virtual ~BinaryElementWiseCannPreparation () {
    for (auto desc : inputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto desc : outputDesc_) {
      aclDestroyTensorDesc(desc);
    }

    for (auto buf : inputBuffers_) {
      aclDestroyDataBuffer(buf);
    }

    for (auto buf : outputBuffers_) {
      aclDestroyDataBuffer(buf);
    }

    aclopDestroyAttr(opAttr_);
  }

  std::vector<aclDataBuffer *> inputBuffers_;
  std::vector<aclDataBuffer *> outputBuffers_;
  std::vector<aclTensorDesc *> inputDesc_;
  std::vector<aclTensorDesc *> outputDesc_;
  aclopAttr *opAttr_;
};

template <typename T>
class Add final : public CannKernel {
 public:
  Add(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public CannKernel {
 public:
  Sub(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public CannKernel {
 public:
  Mul(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public CannKernel {
 public:
  Div(const OpKernelInfo& info) : CannKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cann
}  // namespace onnxruntime
```

binary_elementwise_ops.cc
```c++
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/math/binary_elementwise_ops.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Prepare(OpKernelContext* ctx, BinaryElementWiseCannPreparation* p, std::string onnxType) {
  const aclDataType aclType = getACLType(onnxType);
  aclFormat format = ACL_FORMAT_ND;

  const Tensor* lhs_tensor = ctx->Input<Tensor>(0);
  const Tensor* rhs_tensor = ctx->Input<Tensor>(1);
  Tensor* output_tensor = ctx->Output(0, lhs_tensor->Shape());

  p->inputDesc_.push_back(aclCreateTensorDesc(aclType, lhs_tensor->Shape().NumDimensions(), lhs_tensor->Shape().GetDims().data(), format));
  p->inputDesc_.push_back(aclCreateTensorDesc(aclType, lhs_tensor->Shape().NumDimensions(), lhs_tensor->Shape().GetDims().data(), format));
  p->outputDesc_.push_back(aclCreateTensorDesc(aclType, lhs_tensor->Shape().NumDimensions(), lhs_tensor->Shape().GetDims().data(), format));

  p->inputBuffers_.push_back(aclCreateDataBuffer(const_cast<T*>(lhs_tensor->template Data<T>()), lhs_tensor->SizeInBytes()));
  p->inputBuffers_.push_back(aclCreateDataBuffer(const_cast<T*>(rhs_tensor->template Data<T>()), rhs_tensor->SizeInBytes()));
  p->outputBuffers_.push_back(aclCreateDataBuffer(output_tensor->template MutableData<T>(), output_tensor->SizeInBytes()));

  return Status::OK();
}

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                   \
  template <>                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                           \
    BinaryElementWiseCannPreparation prepare;                                              \
    ORT_RETURN_IF_ERROR(Prepare<T>(context, &prepare, #T));                                \
    CANN_CALL_THROW(aclopCompileAndExecute(#x,                                             \
                                  prepare.inputDesc_.size(),                               \
                                  prepare.inputDesc_.data(),                               \
                                  prepare.inputBuffers_.data(),                            \
                                  prepare.outputDesc_.size(),                              \
                                  prepare.outputDesc_.data(),                              \
                                  prepare.outputBuffers_.data(),                           \
                                  prepare.opAttr_,                                         \
                                  ACL_ENGINE_SYS,                                          \
                                  ACL_COMPILE_SYS,                                         \
                                  NULL,                                                    \
                                  Stream()));                                              \
    return Status::OK();                                                                   \
  }

#define BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(x, startver, endver, T)         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      x,                                                                                   \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      x<T>);

#define BINARY_OP_VERSIONED_TYPED(name, startver, endver, T)                               \
  BINARY_ELEMENTWISE_REGISTER_KERNEL_VERSIONED_TYPED(name, startver, endver, T)            \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

// the postfix of means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#define BINARY_OP_VERSIONED_BCSILHD(name, startver, endver)   \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int8_t)   \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int16_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint8_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, double) 

#define BINARY_OP_VERSIONED_IH(name, startver, endver)        \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)

#define BINARY_OP_VERSIONED_BCSIH(name, startver, endver)     \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, uint8_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int8_t)   \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int16_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)

#define BINARY_OP_VERSIONED_ILHD(name, startver, endver)      \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int32_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, int64_t)  \
  BINARY_OP_VERSIONED_TYPED(name, startver, endver, double)

BINARY_OP_VERSIONED_BCSILHD(Add, 8, 13)
BINARY_OP_VERSIONED_IH(Sub, 8, 13)
BINARY_OP_VERSIONED_BCSIH(Mul, 8, 13)
BINARY_OP_VERSIONED_ILHD(Div, 8, 13)

}  // namespace cann
}  // namespace onnxruntime
```
