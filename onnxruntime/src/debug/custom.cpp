#include "custom.h"
#include "core/common/common.h"

void AddKernel::Compute(OrtKernelContext* context)
{
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const int64_t* in_X = ort_.GetTensorData<int64_t>(input_X);
  const int64_t* in_Y = ort_.GetTensorData<int64_t>(input_Y);

  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  int64_t* out = ort_.GetTensorMutableData<int64_t>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  ORT_UNUSED_PARAMETER(compute_stream_);
  for (int64_t i = 0; i < size; i++) {
    out[i] = in_X[i] + *in_Y + 1;
  }
}
