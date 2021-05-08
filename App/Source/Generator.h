#pragma once

#include <torch/torch.h>
#include <torch/script.h>


const int NUM_SAMPLES = 131072;


torch::jit::script::Module loadModule();

