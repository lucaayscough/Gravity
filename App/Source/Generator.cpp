#include "Generator.h"


Generator::Generator(){
    // Loads generator model.
    module = torch::jit::load("C:\\Program Files\\AdversarialAudio\\traced_generator.pt");
}

void Generator::generateSample(){
    {
        torch::NoGradGuard no_grad;

        // Create random input tensor.
        inputs.clear();
        inputs.push_back(torch::randn({1, 512}));
    
        // Forward input to module.
        output = module.forward(inputs).toTensor();
    }

    // Copy tensor to array.
    for(int i = 0; i < NUM_SAMPLES; i++){
        sound[i] = output[0][0][i].item<float>();
    }
}


