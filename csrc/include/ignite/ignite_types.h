#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#include <vector>

using optim_param_groups = std::vector<torch::optim::OptimizerParamGroup>;
using optim_param_group = torch::optim::OptimizerParamGroup;

using optim_sgd = torch::optim::SGD*;
using optim_adam = torch::optim::Adam*;
using optim_adamw = torch::optim::AdamW*;
using optim_adagrad = torch::optim::Adagrad*;
using optim_rmsprop = torch::optim::RMSprop*;
using script_module = torch::jit::script::Module*;
using torch_stack = torch::jit::Stack*;

using optim_adamw_state = torch::optim::AdamWParamState*;

struct adamw_param_group {
    std::vector<torch::Tensor*> params;
    double lr;
    double weight_decay;
    double momentum;      // Momentum factor
    double dampening;    // Dampening for momentum
    bool nesterov;       // Enables Nesterov momentum

    sgd_param_group(
        std::vector<torch::Tensor*> params,
        double learning_rate,
        double weight_decay,
        double momentum,
        double dampening,
        bool nesterov
    )
        : params(params), lr(learning_rate), weight_decay(weight_decay), momentum(momentum), dampening(dampening), nesterov(nesterov) {}

    sgd_param_group(
        torch::optim::OptimizerParamGroup& group
    ) {
        std::vector<torch::Tensor*> params;
        for (torch::Tensor& param : group.params()) {
            params.push_back(&param);
        }
        auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
        lr = options.lr();
        weight_decay = options.weight_decay();
        momentum = options.momentum();
        dampening = options.dampening();
        nesterov = options.nesterov();
    }

    torch::optim::SGDOptions to_sgd_options() const {
        auto options = torch::optim::SGDOptions(lr);
        options.weight_decay(weight_decay);
        options.momentum(momentum);
        options.dampening(dampening);
        options.nesterov(nesterov);
        return options;
    }
};

using sgd_param_groups = std::vector<sgd_param_group>;


// casting to and from raw pointers
namespace make_raw {
    void* SGD(const optim_sgd& x);
    void* Adam(const optim_adam& x);
    void* AdamW(const optim_adamw& x);
    void* Adagrad(const optim_adagrad& x);
    void* RMSprop(const optim_rmsprop& x);
    void* ScriptModule(const script_module& x);
    void* TorchStack(const torch_stack& x);
    void* OptimParamGroups(const optim_param_groups& x);
    void* OptimParamGroup(const optim_param_group& x);
    void* SGDParamGroups(const sgd_param_groups& x);
    void* SGDParamGroup(const sgd_param_group& x);
}

namespace from_raw {
    optim_sgd SGD(void* x);
    optim_adam Adam(void* x);
    optim_adamw AdamW(void* x);
    optim_adagrad Adagrad(void* x);
    optim_rmsprop RMSprop(void* x);
    script_module ScriptModule(void* x);
    torch_stack TorchStack(void* x);
    optim_param_groups OptimParamGroups(void* x);
    optim_param_group OptimParamGroup(void* x);
    sgd_param_groups SGDParamGroups(void* x);
    sgd_param_group SGDParamGroup(void* x);
}
