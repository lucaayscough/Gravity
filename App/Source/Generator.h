#pragma once


class Generator{
public:
    static const int M_NUM_SAMPLES = 131072;
    static torch::jit::script::Module generator_module, mapper_module;

    // Constructors and destructors.
    Generator();
    ~Generator();

    static at::Tensor generateLatents();
    static juce::Array<float> generateSample(at::Tensor&);
};