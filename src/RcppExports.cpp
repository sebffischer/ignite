// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "ignite_types.h"
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_ignite_sgd_get_param_groups
ignite::sgd_param_groups rcpp_ignite_sgd_get_param_groups(ignite::optim_sgd opt);
RcppExport SEXP _ignite_rcpp_ignite_sgd_get_param_groups(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_sgd >::type opt(optSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_sgd_get_param_groups(opt));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_sgd_set_param_groups
void rcpp_ignite_sgd_set_param_groups(ignite::optim_sgd opt, ignite::sgd_param_groups param_groups);
RcppExport SEXP _ignite_rcpp_ignite_sgd_set_param_groups(SEXP optSEXP, SEXP param_groupsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_sgd >::type opt(optSEXP);
    Rcpp::traits::input_parameter< ignite::sgd_param_groups >::type param_groups(param_groupsSEXP);
    rcpp_ignite_sgd_set_param_groups(opt, param_groups);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adamw_get_param_groups
ignite::adamw_param_groups rcpp_ignite_adamw_get_param_groups(ignite::optim_adamw opt);
RcppExport SEXP _ignite_rcpp_ignite_adamw_get_param_groups(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_get_param_groups(opt));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adamw_set_param_groups
void rcpp_ignite_adamw_set_param_groups(ignite::optim_adamw opt, ignite::adamw_param_groups param_groups);
RcppExport SEXP _ignite_rcpp_ignite_adamw_set_param_groups(SEXP optSEXP, SEXP param_groupsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    Rcpp::traits::input_parameter< ignite::adamw_param_groups >::type param_groups(param_groupsSEXP);
    rcpp_ignite_adamw_set_param_groups(opt, param_groups);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adamw_states
ignite::adamw_states rcpp_ignite_adamw_states(ignite::optim_adamw opt);
RcppExport SEXP _ignite_rcpp_ignite_adamw_states(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_states(opt));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_adamw_state_exp_avg
torch::Tensor rcpp_adamw_state_exp_avg(ignite::adamw_state state);
RcppExport SEXP _ignite_rcpp_adamw_state_exp_avg(SEXP stateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_state >::type state(stateSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_adamw_state_exp_avg(state));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_opt_step
torch::TensorList rcpp_ignite_opt_step(Rcpp::XPtr<XPtrTorchScriptModule> network, Rcpp::XPtr<XPtrTorchScriptModule> loss_fn, XPtrTorchStack input, torch::Tensor target, ignite::optim_sgd optimizer);
RcppExport SEXP _ignite_rcpp_ignite_opt_step(SEXP networkSEXP, SEXP loss_fnSEXP, SEXP inputSEXP, SEXP targetSEXP, SEXP optimizerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<XPtrTorchScriptModule> >::type network(networkSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<XPtrTorchScriptModule> >::type loss_fn(loss_fnSEXP);
    Rcpp::traits::input_parameter< XPtrTorchStack >::type input(inputSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type target(targetSEXP);
    Rcpp::traits::input_parameter< ignite::optim_sgd >::type optimizer(optimizerSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_opt_step(network, loss_fn, input, target, optimizer));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_predict_step
torch::Tensor rcpp_ignite_predict_step(Rcpp::XPtr<XPtrTorchScriptModule> network, XPtrTorchStack input);
RcppExport SEXP _ignite_rcpp_ignite_predict_step(SEXP networkSEXP, SEXP inputSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<XPtrTorchScriptModule> >::type network(networkSEXP);
    Rcpp::traits::input_parameter< XPtrTorchStack >::type input(inputSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_predict_step(network, input));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_sgd
ignite::optim_sgd rcpp_ignite_sgd(torch::TensorList params, double lr, double momentum, double dampening, double weight_decay, bool nesterov);
RcppExport SEXP _ignite_rcpp_ignite_sgd(SEXP paramsSEXP, SEXP lrSEXP, SEXP momentumSEXP, SEXP dampeningSEXP, SEXP weight_decaySEXP, SEXP nesterovSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type momentum(momentumSEXP);
    Rcpp::traits::input_parameter< double >::type dampening(dampeningSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type nesterov(nesterovSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_sgd(params, lr, momentum, dampening, weight_decay, nesterov));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_sgd_step
void rcpp_ignite_sgd_step(ignite::optim_sgd opt);
RcppExport SEXP _ignite_rcpp_ignite_sgd_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_sgd >::type opt(optSEXP);
    rcpp_ignite_sgd_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_sgd_zero_grad
void rcpp_ignite_sgd_zero_grad(ignite::optim_sgd opt);
RcppExport SEXP _ignite_rcpp_ignite_sgd_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_sgd >::type opt(optSEXP);
    rcpp_ignite_sgd_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adam
ignite::optim_adam rcpp_ignite_adam(torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
RcppExport SEXP _ignite_rcpp_ignite_adam(SEXP paramsSEXP, SEXP lrSEXP, SEXP beta1SEXP, SEXP beta2SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type beta1(beta1SEXP);
    Rcpp::traits::input_parameter< double >::type beta2(beta2SEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< bool >::type amsgrad(amsgradSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adam(params, lr, beta1, beta2, eps, weight_decay, amsgrad));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adam_step
void rcpp_ignite_adam_step(ignite::optim_adam opt);
RcppExport SEXP _ignite_rcpp_ignite_adam_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adam >::type opt(optSEXP);
    rcpp_ignite_adam_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adam_zero_grad
void rcpp_ignite_adam_zero_grad(ignite::optim_adam opt);
RcppExport SEXP _ignite_rcpp_ignite_adam_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adam >::type opt(optSEXP);
    rcpp_ignite_adam_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adamw
ignite::optim_adamw rcpp_ignite_adamw(ignite::adamw_param_groups groups);
RcppExport SEXP _ignite_rcpp_ignite_adamw(SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_param_groups >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw(groups));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adamw_step
void rcpp_ignite_adamw_step(ignite::optim_adamw opt);
RcppExport SEXP _ignite_rcpp_ignite_adamw_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    rcpp_ignite_adamw_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adamw_zero_grad
void rcpp_ignite_adamw_zero_grad(ignite::optim_adamw opt);
RcppExport SEXP _ignite_rcpp_ignite_adamw_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    rcpp_ignite_adamw_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adagrad
ignite::optim_adagrad rcpp_ignite_adagrad(torch::TensorList params, double lr, double lr_decay, double weight_decay, double initial_accumulator_value, double eps);
RcppExport SEXP _ignite_rcpp_ignite_adagrad(SEXP paramsSEXP, SEXP lrSEXP, SEXP lr_decaySEXP, SEXP weight_decaySEXP, SEXP initial_accumulator_valueSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type lr_decay(lr_decaySEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< double >::type initial_accumulator_value(initial_accumulator_valueSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adagrad(params, lr, lr_decay, weight_decay, initial_accumulator_value, eps));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adagrad_step
void rcpp_ignite_adagrad_step(ignite::optim_adagrad opt);
RcppExport SEXP _ignite_rcpp_ignite_adagrad_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adagrad >::type opt(optSEXP);
    rcpp_ignite_adagrad_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adagrad_zero_grad
void rcpp_ignite_adagrad_zero_grad(ignite::optim_adagrad opt);
RcppExport SEXP _ignite_rcpp_ignite_adagrad_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adagrad >::type opt(optSEXP);
    rcpp_ignite_adagrad_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_rmsprop
ignite::optim_rmsprop rcpp_ignite_rmsprop(torch::TensorList params, double lr, double alpha, double eps, double weight_decay, double momentum, bool centered);
RcppExport SEXP _ignite_rcpp_ignite_rmsprop(SEXP paramsSEXP, SEXP lrSEXP, SEXP alphaSEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP momentumSEXP, SEXP centeredSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::TensorList >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< double >::type lr(lrSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type weight_decay(weight_decaySEXP);
    Rcpp::traits::input_parameter< double >::type momentum(momentumSEXP);
    Rcpp::traits::input_parameter< bool >::type centered(centeredSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_rmsprop(params, lr, alpha, eps, weight_decay, momentum, centered));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_rmsprop_step
void rcpp_ignite_rmsprop_step(ignite::optim_rmsprop opt);
RcppExport SEXP _ignite_rcpp_ignite_rmsprop_step(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_rmsprop >::type opt(optSEXP);
    rcpp_ignite_rmsprop_step(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_rmsprop_zero_grad
void rcpp_ignite_rmsprop_zero_grad(ignite::optim_rmsprop opt);
RcppExport SEXP _ignite_rcpp_ignite_rmsprop_zero_grad(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_rmsprop >::type opt(optSEXP);
    rcpp_ignite_rmsprop_zero_grad(opt);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_sgd
void rcpp_delete_optim_sgd(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_sgd(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_sgd(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_adam
void rcpp_delete_optim_adam(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_adam(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_adam(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_adamw
void rcpp_delete_optim_adamw(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_adamw(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_adamw(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_adagrad
void rcpp_delete_optim_adagrad(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_adagrad(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_adagrad(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_rmsprop
void rcpp_delete_optim_rmsprop(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_rmsprop(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_rmsprop(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_param_groups
void rcpp_delete_optim_param_groups(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_param_groups(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_param_groups(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_optim_param_group
void rcpp_delete_optim_param_group(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim_param_group(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim_param_group(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_sgd_param_groups
void rcpp_delete_sgd_param_groups(void* x);
RcppExport SEXP _ignite_rcpp_delete_sgd_param_groups(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_sgd_param_groups(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_sgd_param_group
void rcpp_delete_sgd_param_group(void* x);
RcppExport SEXP _ignite_rcpp_delete_sgd_param_group(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_sgd_param_group(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_adamw_param_groups
void rcpp_delete_adamw_param_groups(void* x);
RcppExport SEXP _ignite_rcpp_delete_adamw_param_groups(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_adamw_param_groups(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_adamw_param_group
void rcpp_delete_adamw_param_group(void* x);
RcppExport SEXP _ignite_rcpp_delete_adamw_param_group(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_adamw_param_group(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_adamw_states
void rcpp_delete_adamw_states(void* x);
RcppExport SEXP _ignite_rcpp_delete_adamw_states(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_adamw_states(x);
    return R_NilValue;
END_RCPP
}
// rcpp_delete_adamw_state
void rcpp_delete_adamw_state(void* x);
RcppExport SEXP _ignite_rcpp_delete_adamw_state(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_adamw_state(x);
    return R_NilValue;
END_RCPP
}
// ignite_raise_exception
void ignite_raise_exception();
RcppExport SEXP _ignite_ignite_raise_exception() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    ignite_raise_exception();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ignite_rcpp_ignite_sgd_get_param_groups", (DL_FUNC) &_ignite_rcpp_ignite_sgd_get_param_groups, 1},
    {"_ignite_rcpp_ignite_sgd_set_param_groups", (DL_FUNC) &_ignite_rcpp_ignite_sgd_set_param_groups, 2},
    {"_ignite_rcpp_ignite_adamw_get_param_groups", (DL_FUNC) &_ignite_rcpp_ignite_adamw_get_param_groups, 1},
    {"_ignite_rcpp_ignite_adamw_set_param_groups", (DL_FUNC) &_ignite_rcpp_ignite_adamw_set_param_groups, 2},
    {"_ignite_rcpp_ignite_adamw_states", (DL_FUNC) &_ignite_rcpp_ignite_adamw_states, 1},
    {"_ignite_rcpp_adamw_state_exp_avg", (DL_FUNC) &_ignite_rcpp_adamw_state_exp_avg, 1},
    {"_ignite_rcpp_ignite_opt_step", (DL_FUNC) &_ignite_rcpp_ignite_opt_step, 5},
    {"_ignite_rcpp_ignite_predict_step", (DL_FUNC) &_ignite_rcpp_ignite_predict_step, 2},
    {"_ignite_rcpp_ignite_sgd", (DL_FUNC) &_ignite_rcpp_ignite_sgd, 6},
    {"_ignite_rcpp_ignite_sgd_step", (DL_FUNC) &_ignite_rcpp_ignite_sgd_step, 1},
    {"_ignite_rcpp_ignite_sgd_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_sgd_zero_grad, 1},
    {"_ignite_rcpp_ignite_adam", (DL_FUNC) &_ignite_rcpp_ignite_adam, 7},
    {"_ignite_rcpp_ignite_adam_step", (DL_FUNC) &_ignite_rcpp_ignite_adam_step, 1},
    {"_ignite_rcpp_ignite_adam_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_adam_zero_grad, 1},
    {"_ignite_rcpp_ignite_adamw", (DL_FUNC) &_ignite_rcpp_ignite_adamw, 1},
    {"_ignite_rcpp_ignite_adamw_step", (DL_FUNC) &_ignite_rcpp_ignite_adamw_step, 1},
    {"_ignite_rcpp_ignite_adamw_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_adamw_zero_grad, 1},
    {"_ignite_rcpp_ignite_adagrad", (DL_FUNC) &_ignite_rcpp_ignite_adagrad, 6},
    {"_ignite_rcpp_ignite_adagrad_step", (DL_FUNC) &_ignite_rcpp_ignite_adagrad_step, 1},
    {"_ignite_rcpp_ignite_adagrad_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_adagrad_zero_grad, 1},
    {"_ignite_rcpp_ignite_rmsprop", (DL_FUNC) &_ignite_rcpp_ignite_rmsprop, 7},
    {"_ignite_rcpp_ignite_rmsprop_step", (DL_FUNC) &_ignite_rcpp_ignite_rmsprop_step, 1},
    {"_ignite_rcpp_ignite_rmsprop_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_rmsprop_zero_grad, 1},
    {"_ignite_rcpp_delete_optim_sgd", (DL_FUNC) &_ignite_rcpp_delete_optim_sgd, 1},
    {"_ignite_rcpp_delete_optim_adam", (DL_FUNC) &_ignite_rcpp_delete_optim_adam, 1},
    {"_ignite_rcpp_delete_optim_adamw", (DL_FUNC) &_ignite_rcpp_delete_optim_adamw, 1},
    {"_ignite_rcpp_delete_optim_adagrad", (DL_FUNC) &_ignite_rcpp_delete_optim_adagrad, 1},
    {"_ignite_rcpp_delete_optim_rmsprop", (DL_FUNC) &_ignite_rcpp_delete_optim_rmsprop, 1},
    {"_ignite_rcpp_delete_optim_param_groups", (DL_FUNC) &_ignite_rcpp_delete_optim_param_groups, 1},
    {"_ignite_rcpp_delete_optim_param_group", (DL_FUNC) &_ignite_rcpp_delete_optim_param_group, 1},
    {"_ignite_rcpp_delete_sgd_param_groups", (DL_FUNC) &_ignite_rcpp_delete_sgd_param_groups, 1},
    {"_ignite_rcpp_delete_sgd_param_group", (DL_FUNC) &_ignite_rcpp_delete_sgd_param_group, 1},
    {"_ignite_rcpp_delete_adamw_param_groups", (DL_FUNC) &_ignite_rcpp_delete_adamw_param_groups, 1},
    {"_ignite_rcpp_delete_adamw_param_group", (DL_FUNC) &_ignite_rcpp_delete_adamw_param_group, 1},
    {"_ignite_rcpp_delete_adamw_states", (DL_FUNC) &_ignite_rcpp_delete_adamw_states, 1},
    {"_ignite_rcpp_delete_adamw_state", (DL_FUNC) &_ignite_rcpp_delete_adamw_state, 1},
    {"_ignite_ignite_raise_exception", (DL_FUNC) &_ignite_ignite_raise_exception, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_ignite(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
