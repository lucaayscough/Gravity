#pragma once


class Generator{
public:
    static const int M_NUM_SAMPLES = 131072;

    torch::jit::script::Module generator_module, mapper_module;

    Generator();
    at::Tensor generateLatents();
    juce::Array<float> generateSample(at::Tensor&);
};