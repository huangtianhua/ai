#include <stdio.h>
#include <assert.h>
#include "custom.h"

const static OrtApi* g_ort = NULL;

#define ORT_ABORT_ON_ERROR(expr)                                \
do {                                                            \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
        const char* msg = g_ort->GetErrorMessage(onnx_status);  \
        fprintf(stderr, "%s\n", msg);                           \
        g_ort->ReleaseStatus(onnx_status);                      \
        abort();                                                \
    }                                                           \
} while (0);

static void usage()
{
    printf("usage: <model_path>\n");
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        usage();
        return -1;
    }

    char* model_path = argv[1];
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    OrtEnv* env;
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "reshape", &env));

    AddOp add_op{"CPUExecutionProvider", nullptr};
    Ort::CustomOpDomain cpu_op_domain("");
    cpu_op_domain.Add(&add_op);

    OrtSessionOptions* session_options;
    ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

    //ORT_ABORT_ON_ERROR(g_ort->AddCustomOpDomain(session_options, cpu_op_domain));
    ORT_ABORT_ON_ERROR(g_ort->SetIntraOpNumThreads(session_options, 1));
    ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));

    OrtSession* session;
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

    OrtAllocator* allocator;
    ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t input_num;
    size_t output_num;
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &input_num));
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &output_num));
    printf("The nums of input and output is %d,%d\n", input_num,output_num);

    char** input_names = (char**)malloc(sizeof(char*) * input_num);
    char** output_names = (char**)malloc(sizeof(char*) * output_num);
    int64_t* input_shape = NULL;
    int64_t* output_shape = NULL;
    size_t input_dim;
    size_t output_dim;

    for (int i = 0; i < input_num; i ++) {
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, allocator, &input_names[i]));
        printf("name is %s\n",input_names[i]);

        OrtTypeInfo* typeinfo;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));

        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

        ONNXTensorElementDataType type;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorElementType(tensor_info, &type));
        ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(tensor_info, &input_dim));

        input_shape = (int64_t*)malloc(sizeof(int64_t) * input_dim);
        ORT_ABORT_ON_ERROR(g_ort->GetDimensions(tensor_info, input_shape, input_dim));
        printf("%d,%d\n",*input_shape,input_dim);
    }

    for (int i = 0; i < output_num; i ++) {
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, allocator, &output_names[i]));

        OrtTypeInfo* typeinfo;
        ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));

        const OrtTensorTypeAndShapeInfo* tensor_info;
        ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

        ONNXTensorElementDataType type;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorElementType(tensor_info, &type));
        ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(tensor_info, &output_dim));

        output_shape = (int64_t*)malloc(sizeof(int64_t) * output_dim);
        ORT_ABORT_ON_ERROR(g_ort->GetDimensions(tensor_info, output_shape, output_dim));
    }

    size_t input_tensor_size = input_dim * *input_shape;
    int64_t * input_tensor_values = (int64_t*)malloc(sizeof(int64_t) * input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; i ++)
        input_tensor_values[i] = i + 1;

    OrtMemoryInfo* memory_info;
    ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    OrtValue* input_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)input_tensor_values, input_tensor_size * sizeof(int64_t), input_shape, input_dim, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor));
    g_ort->ReleaseMemoryInfo(memory_info);

    OrtValue* output_tensor = NULL;
    ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, (const char * const*)input_names, (const OrtValue* const*)&input_tensor, 1, (const char * const*)output_names, 1, &output_tensor));

    int64_t* output_tensor_values;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_values));

    for(int i = 0; i < input_tensor_size; i ++)
        printf("%d\n",output_tensor_values[i]);

    return 0;
}
