#include "Generator.h"


torch::jit::script::Module loadModule(){
    torch::jit::script::Module module;
    module = torch::jit::load("traced_generator.pt");
    return module;
}
