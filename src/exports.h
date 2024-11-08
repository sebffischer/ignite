// Generated by using torchexport::export() -> do not edit by hand
#include <Rcpp.h>
#include <torch.h>
#include "ignite_types.h"

ignite::sgd_param_groups rcpp_ignite_sgd_get_param_groups (ignite::optim_sgd opt);
void rcpp_ignite_sgd_set_param_groups (ignite::optim_sgd opt, ignite::sgd_param_groups param_groups);
ignite::adamw_param_groups rcpp_ignite_adamw_get_param_groups (ignite::optim_adamw opt);
void rcpp_ignite_adamw_set_param_groups (ignite::optim_adamw opt, ignite::adamw_param_groups param_groups);
torch::TensorList rcpp_ignite_opt_step (Rcpp::XPtr<XPtrTorchScriptModule> network, Rcpp::XPtr<XPtrTorchScriptModule> loss_fn, XPtrTorchStack input, torch::Tensor target, ignite::optim_sgd optimizer);
torch::Tensor rcpp_ignite_predict_step (Rcpp::XPtr<XPtrTorchScriptModule> network, XPtrTorchStack input);
ignite::optim_sgd rcpp_ignite_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
void rcpp_ignite_sgd_step (ignite::optim_sgd opt);
void rcpp_ignite_sgd_zero_grad (ignite::optim_sgd opt);
ignite::optim_adam rcpp_ignite_adam (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
void rcpp_ignite_adam_step (ignite::optim_adam opt);
void rcpp_ignite_adam_zero_grad (ignite::optim_adam opt);
ignite::optim_adamw rcpp_ignite_adamw (torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
void rcpp_ignite_adamw_step (ignite::optim_adamw opt);
void rcpp_ignite_adamw_zero_grad (ignite::optim_adamw opt);
ignite::optim_adagrad rcpp_ignite_adagrad (torch::TensorList params, double lr, double lr_decay, double weight_decay, double initial_accumulator_value, double eps);
void rcpp_ignite_adagrad_step (ignite::optim_adagrad opt);
void rcpp_ignite_adagrad_zero_grad (ignite::optim_adagrad opt);
ignite::optim_rmsprop rcpp_ignite_rmsprop (torch::TensorList params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered);
void rcpp_ignite_rmsprop_step (ignite::optim_rmsprop opt);
void rcpp_ignite_rmsprop_zero_grad (ignite::optim_rmsprop opt);
void rcpp_delete_optim_sgd (void* x);
void rcpp_delete_optim_adam (void* x);
void rcpp_delete_optim_adamw (void* x);
void rcpp_delete_optim_adagrad (void* x);
void rcpp_delete_optim_rmsprop (void* x);
void rcpp_delete_optim_param_groups (void* x);
void rcpp_delete_optim_param_group (void* x);
void rcpp_delete_sgd_param_groups (void* x);
void rcpp_delete_sgd_param_group (void* x);
void rcpp_delete_adamw_param_groups (void* x);
void rcpp_delete_adamw_param_group (void* x);
