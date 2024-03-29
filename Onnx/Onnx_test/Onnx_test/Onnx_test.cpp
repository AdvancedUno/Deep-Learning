﻿
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Helpers.cpp"
//#include "cuda_provider_factory.h"
//#include "onnxruntime_cxx_api.h"
//#include "tensorrt_provider_factory.h"
//#include "provider_options.h"




#include "D:\Wintec\onnxruntime\onnxruntime\core\providers\cuda/cuda_provider_factory.h"
#include "../packages/Microsoft.ML.OnnxRuntime.Gpu.1.11.0/build/native/include/onnxruntime_cxx_api.h"
//#include "../packages/Microsoft.ML.OnnxRuntime.Gpu.1.11.0/build/native/include/tensorrt_provider_factory.h"

#include <chrono>

//#include "cuda_provider_factory.h"

using namespace std;


int main()
{
    clock_t start;







    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 1;
    constexpr int64_t width = 512;
    constexpr int64_t height = 512;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;

    const string imageFile = "C:/Users/sales/source/repos/Onnx_test/Onnx_test/repos/f3.bmp";
    //const string labelFile = "C:/Users/sales/source/repos/Onnx_test/Onnx_test/repos/f2.bmp";
    auto modelPath = L"C:/Users/sales/source/repos/Onnx_test/Onnx_test/repos/UnoDenoise.onnx";

    // load labels 
   /* vector<float> labels = loadLabels(labelFile);
    if (labels.empty()) {
        cout << "Failed to load" << imageFile << endl;
        return 1;
    }*/



    // load image
    const vector<float> imageVec = loadImage(imageFile);
    if (imageVec.empty()) {
        cout << "Failed to load image : " << imageFile << endl;
        return 1;
    }

    cout << imageVec.size() << endl;

    if (imageVec.size() != numInputElements) {
        cout << "Invalid image format. " << endl;
        return 1;
    }



    /* Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
         "efficient_Unet");
     Ort::SessionOptions sessionOptions;
     sessionOptions.SetIntraOpNumThreads(1);*/
     //OrtCUDAProviderOptions cuda_options ;
     //sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
     //sessionOptions.SetGraphOptimizationLevel(
     //    GraphOptimizationLevel::ORT_ENABLE_EXTENDED);


    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);





    Ort::SessionOptions session_options;
    // cuda 사용하기 위해 provider 추가
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    // 추가 graph 최적화 옵션 
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);



    //Ort::Env env = Ort::Env{ ORT_LOGGING_LEVEL_ERROR, "Default" };
    //Ort::SessionOptions sf;
    //int device_id = 0;
    //
    //OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
    //Ort::Session session(env, modelPath, sf);


    for (int i = 0; i < 10; i++) {


        start = clock();

        // create Session
        //session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });
        session = Ort::Session(env, modelPath, session_options);


        // Define I/O array
        // We can use vector instead of array
        const array<int64_t, 4> inputShape = { 1, numChannels, height, width };
        const array<int64_t, 4> outputShape = { 1, numChannels, height, width };



        // define I/O Tensor
        // It golds the array pointer internally.
        // Dont's delete array while the Tensor alive.
        // If use vector, Dont's reallocate memory after creating the Tensor

        // define array
        array<float, numInputElements> input;
        array<float, numInputElements> results;

        std::vector<const char*> output_node_names = { "squeezenet0_flatten0_reshape0" };

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
        const array<const char*, > outputNames = { outputName };
        ort_alloc.Free(inputName);
        ort_alloc.Free(outputName);


        // run inference
        try {
            session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
        }
        catch (Ort::Exception& e) {
            cout << e.what() << endl;

            return 1;
        }
        cout << "aaaaa" << endl;

        std::cout << "Time: " << (clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    }


    //cout << results.size() << endl;

    //Mat img = cv::Mat(512, 512, CV_32FC1, results.data());


    //cv::imshow("img", img);

    //waitKey(0);



     //sort results
   /* vector<pair<size_t, float>> indexValuePairs;

    for (size_t i = 0; i < results.size(); i++) {
        indexValuePairs.emplace_back(i, results[i]);

    }

    sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) {return lhs.second > rhs.second; });*/


    //for (size_t i = 0; i < 5; ++i) {
    //    const auto& result = indexValuePairs[i];
    //    cout << i + 1 << ": " << labels[result.first] << " " << result.second << endl;
    //}




}
