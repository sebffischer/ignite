#include <torch/torch.h>
#include "ignite/ignite_types.h"
#include <lantern/types.h>

namespace make_raw {

void* SGD(const optim_sgd& x) {
    return make_ptr<optim_sgd>(x);
}
void* Adam(const optim_adam& x) {
    return make_ptr<optim_adam>(x);
}
void* AdamW(const optim_adamw& x) {
    return make_ptr<optim_adamw>(x);
}
void* Adagrad(const optim_adagrad& x) {
    return make_ptr<optim_adagrad>(x);
}
void* RMSprop(const optim_rmsprop& x) {
    return make_ptr<optim_rmsprop>(x);
}
void* ScriptModule(const script_module& x) {
    return make_ptr<script_module>(x);
}
void* TorchStack(const torch_stack& x) {
    return make_ptr<torch_stack>(x);
}
void* OptimParamGroups(const optim_param_groups& x) {
    return make_ptr<optim_param_groups>(x);
}
void* OptimParamGroup(const optim_param_group& x) {
    return make_ptr<optim_param_group>(x);
}
void* SGDParamGroup(const sgd_param_group& x) {
    return make_ptr<sgd_param_group>(x);
}
void* SGDParamGroups(const sgd_param_groups& x) {
    return make_ptr<sgd_param_groups>(x);
}
void* AdamWParamGroup(const adamw_param_group& x) {
    return make_ptr<adamw_param_group>(x);
}
void* AdamWParamGroups(const adamw_param_groups& x) {
    return make_ptr<adamw_param_groups>(x);
}

namespace from_raw {
optim_sgd SGD(void* x) {
    return *reinterpret_cast<optim_sgd*>(x);
}
optim_adam Adam(void* x) {
    return *reinterpret_cast<optim_adam*>(x);
}
optim_adamw AdamW(void* x) {
    return *reinterpret_cast<optim_adamw*>(x);
}
optim_adagrad Adagrad(void* x) {
    return *reinterpret_cast<optim_adagrad*>(x);
}
optim_rmsprop RMSprop(void* x) {
    return *reinterpret_cast<optim_rmsprop*>(x);
}
script_module ScriptModule(void* x) {
    return *reinterpret_cast<script_module*>(x);
}
torch_stack TorchStack(void* x) {
    return reinterpret_cast<torch_stack>(x);
}
optim_param_groups OptimParamGroups(void* x) {
    return *reinterpret_cast<optim_param_groups*>(x);
}
optim_param_group OptimParamGroup(void* x) {
    return *reinterpret_cast<optim_param_group*>(x);
}
sgd_param_groups SGDParamGroups(void* x) {
    return *reinterpret_cast<sgd_param_groups*>(x);
}
sgd_param_group SGDParamGroup(void* x) {
    return *reinterpret_cast<sgd_param_group*>(x);
}
adamw_param_group AdamWParamGroup(void* x) {
    return *reinterpret_cast<adamw_param_group*>(x);
}
adamw_param_groups AdamWParamGroups(void* x) {
    return *reinterpret_cast<adamw_param_groups*>(x);
}
}

// [[torch::export]]
void delete_optim_sgd(void* x) {
  delete reinterpret_cast<optim_sgd>(x);
}
// [[torch::export]]
void delete_optim_adam(void* x) {
  delete reinterpret_cast<optim_adam>(x);
}
// [[torch::export]]
void delete_optim_adamw(void* x) {
  delete reinterpret_cast<optim_adamw>(x);
}
// [[torch::export]]
void delete_optim_adagrad(void* x) {
  delete reinterpret_cast<optim_adagrad>(x);
}
// [[torch::export]]
void delete_optim_rmsprop(void* x) {
  delete reinterpret_cast<optim_rmsprop>(x);
}
// [[torch::export]]
void delete_optim_param_groups(void* x) {
  delete reinterpret_cast<optim_param_groups*>(x);
}
// [[torch::export]]
void delete_optim_param_group(void* x) {
  delete reinterpret_cast<optim_param_group*>(x);
}
// [[torch::export]]
void delete_sgd_param_groups(void* x) {
  delete reinterpret_cast<sgd_param_groups*>(x);
}
// [[torch::export]]
void delete_sgd_param_group(void* x) {
  delete reinterpret_cast<sgd_param_group*>(x);
}
// [[torch::export]]
void delete_adamw_param_groups(void* x) {
  delete reinterpret_cast<adamw_param_groups*>(x);
}
// [[torch::export]]
void delete_adamw_param_group(void* x) {
  delete reinterpret_cast<adamw_param_group*>(x);
}
