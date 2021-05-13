#pragma once

#include <torch/torch.h>
#include <torch/script.h>


const int NUM_SAMPLES = 131072;

class Generator{
public:
    torch::jit::script::Module module;
    float sound[NUM_SAMPLES] = {};
    
    std::vector<torch::jit::IValue> inputs;
    at::Tensor output;

    Generator();
    void generateSample();
};

