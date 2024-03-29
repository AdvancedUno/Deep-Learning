// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "Helpers.cpp"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#ifdef HAVE_TENSORRT_PROVIDER_FACTORY_H
#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>

std::unique_ptr<OrtTensorRTProviderOptionsV2> get_default_trt_provider_options() {
    auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
    tensorrt_options->device_id = 0;
    tensorrt_options->has_user_compute_stream = 0;
    tensorrt_options->user_compute_stream = nullptr;
    tensorrt_options->trt_max_partition_iterations = 1000;
    tensorrt_options->trt_min_subgraph_size = 1;
    tensorrt_options->trt_max_workspace_size = 1 << 30;
    tensorrt_options->trt_fp16_enable = false;
    tensorrt_options->trt_int8_enable = false;
    tensorrt_options->trt_int8_calibration_table_name = "";
    tensorrt_options->trt_int8_use_native_calibration_table = false;
    tensorrt_options->trt_dla_enable = false;
    tensorrt_options->trt_dla_core = 0;
    tensorrt_options->trt_dump_subgraphs = false;
    tensorrt_options->trt_engine_cache_enable = false;
    tensorrt_options->trt_engine_cache_path = "";
    tensorrt_options->trt_engine_decryption_enable = false;
    tensorrt_options->trt_engine_decryption_lib_path = "";
    tensorrt_options->trt_force_sequential_engine_build = false;

    return tensorrt_options;
}
//#endif

void run_ort_trt() {
    Ort::Env env;
    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    const auto& api = Ort::GetApi();

    //const string imageFile = "C:/Users/sales/source/repos/OnnxTest/OnnxTest/repos/f3.bmp";
    const string imageFile = "D:/trash/000.png";

    OrtTensorRTProviderOptionsV2* tensorrt_options;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    const wchar_t* model_path = L"D:/trash/CFA_resnet18.onnx";
    //const wchar_t* model_path = L"C:/Users/sales/source/repos/onnxTest/onnxTest/repos/UnoDenoise.onnx";
#else
    const char* model_path = "squeezenet.onnx";
#endif

    const vector<float> imageVec = loadImage(imageFile, 224, 224);
    //*****************************************************************************************
    // It's not suggested to directly new OrtTensorRTProviderOptionsV2 to get provider options
    //*****************************************************************************************
    //
    // auto tensorrt_options = get_default_trt_provider_options();
    // session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options.get());

    //**************************************************************************************************************************
    // It's suggested to use CreateTensorRTProviderOptions() to get provider options
    // since ORT takes care of valid options for you 
    //**************************************************************************************************************************
    api.CreateTensorRTProviderOptions(&tensorrt_options);
    std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
    api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get());

    printf("Runing ORT TRT EP with default provider options\n");

    Ort::Session session(env, model_path, session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (size_t j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %zu=%jd\n", i, j, input_node_dims[j]);
    }

    size_t input_tensor_size = 3 * 224 * 224;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!

    ///////////////////////////////////////////////////////////////////////////
    const array<int64_t, 4> inputShape = { 1, 3, 224, 224 };
    const array<int64_t, 4> outputShape = { 1, 256, 14, 14 };

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    //constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;
    // define I/O Tensor
    // It golds the array pointer internally.
    // Dont's delete array while the Tensor alive.
    // If use vector, Dont's reallocate memory after creating the Tensor

    // define array
    array<float, numInputElements> input;
    array<float, numInputElements> results;

    std::vector<const char*> output_node_names = { "173" };

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());


    // copy image data to input array
    copy(imageVec.begin(), imageVec.end(), input.begin());


    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* inputName = session.GetInputName(0, ort_alloc);
    char* outputName = session.GetOutputName(0, ort_alloc);
    const array<const char*, 1> inputNames = { inputName };
    const array<const char*, 1> outputNames = { outputName };
    ort_alloc.Free(inputName);
    ort_alloc.Free(outputName);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////






    std::vector<float> input_tensor_values(input_tensor_size);
    //std::vector<const char*> output_node_names = { "output1" };

    if (imageVec.empty()) {
        cout << "Failed to load image : " << imageFile << endl;
        return;
    }
    // initialize input data with values in [0.0, 1.0]
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // create input tensor object from data values
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
   // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), input_tensor_size, input_node_dims.data(), 4);
    //Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), input_tensor_size, input_node_dims.data(), 4);


    //copy(imageVec.begin(), imageVec.end(), input_tensor.begin());


    assert(input_tensor.IsTensor());
    //session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    // score model & input tensor, get back output tensor
    //auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    //auto output_tensors = 
    cout << "nnnnnnnann" << endl;
    Ort::RunOptions runOptions;
    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

    //assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    cout << outputTensor.IsTensor() << endl;
    cout << "aaaaaaaaaaaaa" << endl;


    // Get pointer to output tensor float values
    //float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    //assert(abs(floatarr[0] - 0.000045) < 1e-6);

    //// score the model, and print scores for first 5 classes
    //for (int i = 0; i < 5; i++)
    //    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

    // Results should be as below...
    // Score for class[0] = 0.000045
    // Score for class[1] = 0.003846
    // Score for class[2] = 0.000125
    // Score for class[3] = 0.001180
    // Score for class[4] = 0.001317

    //Mat img = cv::Mat(224, 224, CV_32FC3, results.data());

    //img.convertTo(img, CV_8UC3, 255);

    //cv::imwrite("denoisedImage.bmp", img);

    //cv::imshow("img", img);

   // waitKey(0);
    // release buffers allocated by ORT alloctor
    for (const char* node_name : input_node_names)
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

    printf("Done!\n");
}

int main(int argc, char* argv[]) {
    run_ort_trt();
    return 0;
}