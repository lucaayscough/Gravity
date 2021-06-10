#pragma once


struct ReferenceCountedTensor: public ReferenceCountedObject{
    ReferenceCountedTensor(at::Tensor t):
    tensor(t){}
    at::Tensor tensor;
    using Ptr = ReferenceCountedObjectPtr<ReferenceCountedTensor>;
    at::Tensor& getTensor(){return tensor;}
};


class Generator{
public:
    static const int M_NUM_SAMPLES = 131072;
    static torch::jit::script::Module generator_module, mapper_module;

    // Constructors and destructors.
    Generator();
    ~Generator();

    static at::Tensor generateLatents(std::int64_t);
    static juce::var generateSample(at::Tensor&);
};
