// Generated by using torchexport::export() -> do not edit by hand
#include "exports.h"
#define IGNITE_HEADERS_ONLY
#include <ignite/ignite.h>

// [[Rcpp::export]]
ignite::optim_sgd rcpp_ignite_sgd (torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {
  return  ignite_sgd(params.get(), lr, momentum, dampening, weight_decay, nesterov);
}
// [[Rcpp::export]]
void rcpp_ignite_sgd_step (ignite::optim_sgd opt) {
   ignite_sgd_step(opt.get());
}
// [[Rcpp::export]]
void rcpp_ignite_sgd_zero_grad (ignite::optim_sgd opt) {
   ignite_sgd_zero_grad(opt.get());
}
// [[Rcpp::export]]
torch::Tensor rcpp_ignite_run_script_module (Rcpp::XPtr<XPtrTorchScriptModule> network, Rcpp::XPtr<XPtrTorchFunctionPtr> loss_fn, torch::Tensor input, torch::Tensor target, ignite::optim_sgd optimizer) {
  return  ignite_run_script_module(network.get(), loss_fn.get(), input.get(), target.get(), optimizer.get());
}
// [[Rcpp::export]]
torch::TensorList rcpp_ignite_forward (torch::Tensor input, torch::Tensor weights, torch::Tensor bias, torch::Tensor old_h, torch::Tensor old_cell) {
  return  ignite_forward(input.get(), weights.get(), bias.get(), old_h.get(), old_cell.get());
}
// [[Rcpp::export]]
torch::TensorList rcpp_ignite_backward (torch::Tensor grad_h, torch::Tensor grad_cell, torch::Tensor new_cell, torch::Tensor input_gate, torch::Tensor output_gate, torch::Tensor candidate_cell, torch::Tensor X, torch::Tensor gate_weights, torch::Tensor weights) {
  return  ignite_backward(grad_h.get(), grad_cell.get(), new_cell.get(), input_gate.get(), output_gate.get(), candidate_cell.get(), X.get(), gate_weights.get(), weights.get());
}
// [[Rcpp::export]]
void rcpp_delete_optim_sgd (void* x) {
   delete_optim_sgd(x);
}
// [[Rcpp::export]]
void rcpp_delete_graph_function (void* x) {
   delete_graph_function(x);
}
// [[Rcpp::export]]
void rcpp_delete_script_module2 (void* x) {
   delete_script_module2(x);
}
// [[Rcpp::export]]
void rcpp_delete_stack2 (void* x) {
   delete_stack2(x);
}
