#pragma once

#include <torch/torch.h>
#include <torch/script.h>


const int NUM_SAMPLES = 131072;

class Generator{
public:
    torch::jit::script::Module generator_module, mapper_module;
    float sound[NUM_SAMPLES] = {};

    Generator();
    at::Tensor generateLatents();
    void generateSample(at::Tensor latents);
};

