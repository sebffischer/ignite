// Generated by using torchexport::export() -> do not edit by hand
#include <Rcpp.h>
#include <torch.h>
#include "ignite_types.h"

ignite::adamw_param_groups rcpp_ignite_adamw_get_param_groups (ignite::optim_adamw groups);
torch::TensorList rcpp_ignite_optim_get_param_group_params (ignite::optim_param_group group);
ignite::adamw_options rcpp_ignite_adamw_get_param_group_options (ignite::optim_param_group group);
void rcpp_ignite_adamw_set_param_group_options (ignite::optim_adamw opt, int i, ignite::adamw_options options);
ignite::adamw_states rcpp_ignite_adamw_get_states (ignite::optim_adamw opt);
torch::TensorList rcpp_ignite_adamw_get_state (ignite::adamw_state state);
torch::Tensor rcpp_adamw_state_exp_avg (ignite::adamw_state state);
torch::Tensor rcpp_adamw_state_exp_avg_sq (ignite::adamw_state state);
torch::Tensor rcpp_adamw_state_max_exp_avg_sq (ignite::adamw_state state);
torch::Tensor rcpp_adamw_state_step (ignite::adamw_state state);
torch::TensorList rcpp_ignite_opt_step (Rcpp::XPtr<XPtrTorchScriptModule> network, Rcpp::XPtr<XPtrTorchScriptModule> loss_fn, XPtrTorchStack input, torch::Tensor target, ignite::optim optimizer);
torch::Tensor rcpp_ignite_predict_step (Rcpp::XPtr<XPtrTorchScriptModule> network, XPtrTorchStack input);
ignite::optim_adamw rcpp_ignite_adamw (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
void rcpp_ignite_adamw_step (ignite::optim_adamw opt);
void rcpp_ignite_adamw_zero_grad (ignite::optim_adamw opt);
void rcpp_delete_optim (void* x);
void rcpp_delete_optim_sgd (void* x);
void rcpp_delete_optim_adam (void* x);
void rcpp_delete_optim_adamw (void* x);
void rcpp_delete_optim_adagrad (void* x);
void rcpp_delete_optim_rmsprop (void* x);
void rcpp_delete_optim_param_groups (void* x);
void rcpp_delete_optim_param_group (void* x);
void rcpp_delete_adamw_param_groups (void* x);
void rcpp_delete_adamw_param_group (void* x);
void rcpp_delete_adamw_states (void* x);
void rcpp_delete_adamw_state (void* x);
void rcpp_delete_adamw_options (void* x);
