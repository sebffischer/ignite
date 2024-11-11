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


struct sgd_param_group {
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

struct adamw_param_group {
    std::vector<torch::Tensor*> params;
    double lr;
    double weight_decay;
    std::pair<double, double> betas;
    double eps;
    bool amsgrad;

    adamw_param_group(
        std::vector<torch::Tensor*> params,
        double learning_rate,
        double weight_decay,
        std::pair<double, double> betas,
        double eps,
        bool amsgrad
    )
        : params(params), lr(learning_rate), weight_decay(weight_decay), betas(betas), eps(eps), amsgrad(amsgrad) {}

    adamw_param_group(
        torch::optim::OptimizerParamGroup& group
    ) {
        for (torch::Tensor& param : group.params()) {
            params.push_back(&param);
        }
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        lr = options.lr();
        weight_decay = options.weight_decay();
        betas = options.betas();
        eps = options.eps();
        amsgrad = options.amsgrad();
    }

    torch::optim::OptimizerParamGroup to_adamw_group_params() const {
        std::vector<torch::Tensor> params_vec;
        std::cout << "params length in ignite_adamw: " << params.size() << std::endl;
        for (auto* param : params) {
            std::cout << "dimension of the tensor: " << param->dim() << std::endl;
            std::cout << "Hello" << std::endl;
            params_vec.push_back(*param);
            // print dimension of the tensor
        }
        // print the
        auto options = to_adamw_options();
        return torch::optim::OptimizerParamGroup(params_vec, std::make_unique<torch::optim::AdamWOptions>(options));
    }

    torch::optim::AdamWOptions to_adamw_options() const {
        auto options = torch::optim::AdamWOptions(lr);
        options.weight_decay(weight_decay);
        options.betas(betas);
        options.eps(eps);
        options.amsgrad(amsgrad);
        return options;
    }
};

using adamw_param_groups = std::vector<adamw_param_group>;

struct adamw_state {
    torch::Tensor* exp_avg;
    torch::Tensor* exp_avg_sq;
    torch::Tensor* max_exp_avg_sq;
    int64_t step;

    adamw_state(torch::optim::AdamWParamState& state) {
        exp_avg = &state.exp_avg();
        exp_avg_sq = &state.exp_avg_sq();
        max_exp_avg_sq = &state.max_exp_avg_sq();
        step = state.step();
    }
};

using adamw_states = std::vector<adamw_state>;

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
    void* AdamWParamGroups(const adamw_param_groups& x);
    void* AdamWParamGroup(const adamw_param_group& x);
    void* AdamWStates(const adamw_states& x);
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
    adamw_param_groups AdamWParamGroups(void* x);
    adamw_param_group AdamWParamGroup(void* x);
    adamw_states AdamWStates(void* x);
}
