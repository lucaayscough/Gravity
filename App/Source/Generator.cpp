#include "Headers.h"


Generator::Generator(){
    // Loads generator model.
    generator_module = torch::jit::load("C:\\Program Files\\AdversarialAudio\\generator_module.pt");
    mapper_module = torch::jit::load("C:\\Program Files\\AdversarialAudio\\mapper_module.pt");
}

at::Tensor Generator::generateLatents(){
    torch::NoGradGuard no_grad;

    // Create random input tensor.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 512}));

    // Forward input to module.
    at::Tensor output = mapper_module.forward(inputs).toTensor();

    return output;
}

juce::Array<float> Generator::generateSample(at::Tensor& latents){
    torch::NoGradGuard no_grad;

    // Create input tensor with latents.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(latents);

    // Forward input to module.
    at::Tensor output = generator_module.forward(inputs).toTensor();

    juce::Array<float> sample;
    sample.ensureStorageAllocated(M_NUM_SAMPLES);

    // Copy tensor to array.
    for(int i = 0; i < M_NUM_SAMPLES; i++){
        sample.insert(i, output[0][0][i].item<float>());
    }

    return sample;
}