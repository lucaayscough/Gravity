#include "Headers.h"


torch::jit::script::Module Generator::generator_module = torch::jit::load("/Users/lucaayscough/dev/AdversarialAudio/Generator/scripted_modules/generator_module.pt");
torch::jit::script::Module Generator::mapper_module = torch::jit::load("/Users/lucaayscough/dev/AdversarialAudio/Generator/scripted_modules/mapper_module.pt");


//------------------------------------------------------------//
// Constructors and destructors.

Generator::Generator(){}
Generator::~Generator(){}


at::Tensor Generator::generateLatents(){
    torch::NoGradGuard no_grad;

    // Create random input tensor.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 512}));

    // Forward input to module.
    at::Tensor output = mapper_module.forward(inputs).toTensor();

    return output;
}

juce::var Generator::generateSample(at::Tensor& latents){
    torch::NoGradGuard no_grad;

    // Create input tensor with latents.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(latents);

    // Forward input to module.
    at::Tensor output = generator_module.forward(inputs).toTensor();

    juce::var sample;

    // Copy tensor to array.
    for(int i = 0; i < M_NUM_SAMPLES; i++){
        sample.append(output[0][0][i].item<float>());
    }

    return sample;
}
