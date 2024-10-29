#include <ATen/core/stack.h>
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/randn.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#define LANTERN_TYPES_IMPL // Should be defined only in a single file.
#include <lantern/types.h>
#include <vector>
#include "ignite/ignite.h"
#include "ignite/ignite_types.h"

#include <torch/script.h>  // One-stop header.

// [[torch::export(register_types=c("optim_sgd", "SGD", "void*", "ignite::optim_sgd"))]]
optim_sgd ignite_sgd(torch::TensorList params, double lr, double momentum, double dampening,
                        double weight_decay, bool nesterov) {

  auto options = torch::optim::SGDOptions(lr)
    .momentum(momentum)
    .dampening(dampening)
    .weight_decay(weight_decay)
    .nesterov(nesterov);
 return new torch::optim::SGD(params.vec(), options);
}

// [[torch::export]]
void ignite_sgd_step(optim_sgd opt) {
  opt->step();
}

// [[torch::export]]
void ignite_sgd_zero_grad(optim_sgd opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=c("optim_adam", "Adam", "void*", "ignite::optim_adam"))]]
optim_adam ignite_adam(torch::TensorList params, double lr, double beta1, double beta2,
                      double eps, double weight_decay, bool amsgrad) {
  auto options = torch::optim::AdamOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);
  return new torch::optim::Adam(params.vec(), options);
}

// [[torch::export]]
void ignite_adam_step(optim_adam opt) {
  opt->step();
}

// [[torch::export]]
void ignite_adam_zero_grad(optim_adam opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=c("optim_adamw", "AdamW", "void*", "ignite::optim_adamw"))]]
optim_adamw ignite_adamw(torch::TensorList params, double lr, double beta1, double beta2,
                        double eps, double weight_decay, bool amsgrad) {
  auto options = torch::optim::AdamWOptions(lr)
    .betas(std::make_tuple(beta1, beta2))
    .eps(eps)
    .weight_decay(weight_decay)
    .amsgrad(amsgrad);
  return new torch::optim::AdamW(params.vec(), options);
}

// [[torch::export]]
void ignite_adamw_step(optim_adamw opt) {
  opt->step();
}

// [[torch::export]]
void ignite_adamw_zero_grad(optim_adamw opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=c("optim_adagrad", "Adagrad", "void*", "ignite::optim_adagrad"))]]
optim_adagrad ignite_adagrad(torch::TensorList params, double lr, double lr_decay, double weight_decay,
                            double initial_accumulator_value, double eps) {
  auto options = torch::optim::AdagradOptions(lr)
    .lr_decay(lr_decay)
    .weight_decay(weight_decay)
    .initial_accumulator_value(initial_accumulator_value)
    .eps(eps);
  return new torch::optim::Adagrad(params.vec(), options);
}

// [[torch::export]]
void ignite_adagrad_step(optim_adagrad opt) {
  opt->step();
}

// [[torch::export]]
void ignite_adagrad_zero_grad(optim_adagrad opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=c("optim_rmsprop", "RMSprop", "void*", "ignite::optim_rmsprop"))]]
optim_rmsprop ignite_rmsprop(torch::TensorList params, double lr, double alpha, double eps,
                            double weight_decay, double momentum, bool centered) {
  auto options = torch::optim::RMSpropOptions(lr)
    .alpha(alpha)
    .eps(eps)
    .weight_decay(weight_decay)
    .momentum(momentum)
    .centered(centered);
  return new torch::optim::RMSprop(params.vec(), options);
}

// [[torch::export]]
void ignite_rmsprop_step(optim_rmsprop opt) {
  opt->step();
}

// [[torch::export]]
void ignite_rmsprop_zero_grad(optim_rmsprop opt) {
  opt->zero_grad();
}

// [[torch::export(register_types=list(c("script_module", "ScriptModule", "void*", "Rcpp::XPtr<XPtrTorchScriptModule>"), c("torch_stack", "TorchStack", "void*", "XPtrTorchStack")))]]
std::vector<torch::Tensor> ignite_opt_step(script_module network, script_module loss_fn, torch_stack input, torch::Tensor target, optim_sgd optimizer) {
  optimizer->zero_grad();

  auto out = (*network)(*input);
  auto loss_inputs = new torch::jit::Stack();
  loss_inputs->push_back(out);
  loss_inputs->push_back(target);

  auto loss = (*loss_fn)(*loss_inputs);
  loss.toTensor().backward();
  optimizer->step();

  std::vector<torch::Tensor> result;
  result.push_back(loss.toTensor());
  result.push_back(out.toTensor());
  return result;
}

// [[torch::export]]
torch::Tensor ignite_predict_step(script_module network, torch_stack input) {
  torch::NoGradGuard no_grad;
  auto out = (*network)(*input);
  return out.toTensor();
}


IGNITE_API int _raise_exception ()
{
  try {
    throw std::runtime_error("Error from IGNITE");
  } IGNITE_HANDLE_EXCEPTION
  return 1;
}
