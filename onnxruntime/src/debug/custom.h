#include <core/session/onnxruntime_cxx_api.h>
#include <vector>

struct OrtTensorDimensions : std::vector<int64_t>
{
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct AddKernel
{
  AddKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info, void* compute_stream)
      : ort_(ort), compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* context);

 private:
  Ort::CustomOpApi ort_;
  void* compute_stream_;
};

struct AddOp : Ort::CustomOpBase<AddOp, AddKernel>
{
  explicit AddOp(const char* provider, void* compute_stream) : provider_(provider), compute_stream_(compute_stream) {}

  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new AddKernel(api, info, compute_stream_); };
  const char* GetName() const { return "Add"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };

 private:
  const char* provider_;
  void* compute_stream_;
};
