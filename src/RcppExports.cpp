// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "ignite_types.h"
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_ignite_adamw_get_param_groups
ignite::adamw_param_groups rcpp_ignite_adamw_get_param_groups(ignite::optim_adamw groups);
RcppExport SEXP _ignite_rcpp_ignite_adamw_get_param_groups(SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_get_param_groups(groups));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_optim_get_param_group_params
torch::TensorList rcpp_ignite_optim_get_param_group_params(ignite::optim_param_group group);
RcppExport SEXP _ignite_rcpp_ignite_optim_get_param_group_params(SEXP groupSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_param_group >::type group(groupSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_optim_get_param_group_params(group));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adamw_get_param_group_options
ignite::adamw_options rcpp_ignite_adamw_get_param_group_options(ignite::optim_param_group group);
RcppExport SEXP _ignite_rcpp_ignite_adamw_get_param_group_options(SEXP groupSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_param_group >::type group(groupSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_get_param_group_options(group));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adamw_set_param_group_options
void rcpp_ignite_adamw_set_param_group_options(ignite::optim_adamw opt, int i, ignite::adamw_options options);
RcppExport SEXP _ignite_rcpp_ignite_adamw_set_param_group_options(SEXP optSEXP, SEXP iSEXP, SEXP optionsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< ignite::adamw_options >::type options(optionsSEXP);
    rcpp_ignite_adamw_set_param_group_options(opt, i, options);
    return R_NilValue;
END_RCPP
}
// rcpp_ignite_adamw_get_states
ignite::adamw_states rcpp_ignite_adamw_get_states(ignite::optim_adamw opt);
RcppExport SEXP _ignite_rcpp_ignite_adamw_get_states(SEXP optSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::optim_adamw >::type opt(optSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_get_states(opt));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_adamw_get_state
torch::TensorList rcpp_ignite_adamw_get_state(ignite::adamw_state state);
RcppExport SEXP _ignite_rcpp_ignite_adamw_get_state(SEXP stateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_state >::type state(stateSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw_get_state(state));
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
// rcpp_adamw_state_exp_avg_sq
torch::Tensor rcpp_adamw_state_exp_avg_sq(ignite::adamw_state state);
RcppExport SEXP _ignite_rcpp_adamw_state_exp_avg_sq(SEXP stateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_state >::type state(stateSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_adamw_state_exp_avg_sq(state));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_adamw_state_max_exp_avg_sq
torch::Tensor rcpp_adamw_state_max_exp_avg_sq(ignite::adamw_state state);
RcppExport SEXP _ignite_rcpp_adamw_state_max_exp_avg_sq(SEXP stateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_state >::type state(stateSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_adamw_state_max_exp_avg_sq(state));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_adamw_state_step
torch::Tensor rcpp_adamw_state_step(ignite::adamw_state state);
RcppExport SEXP _ignite_rcpp_adamw_state_step(SEXP stateSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< ignite::adamw_state >::type state(stateSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_adamw_state_step(state));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_ignite_opt_step
torch::TensorList rcpp_ignite_opt_step(Rcpp::XPtr<XPtrTorchScriptModule> network, Rcpp::XPtr<XPtrTorchScriptModule> loss_fn, XPtrTorchStack input, torch::Tensor target, ignite::optim optimizer);
RcppExport SEXP _ignite_rcpp_ignite_opt_step(SEXP networkSEXP, SEXP loss_fnSEXP, SEXP inputSEXP, SEXP targetSEXP, SEXP optimizerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<XPtrTorchScriptModule> >::type network(networkSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<XPtrTorchScriptModule> >::type loss_fn(loss_fnSEXP);
    Rcpp::traits::input_parameter< XPtrTorchStack >::type input(inputSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type target(targetSEXP);
    Rcpp::traits::input_parameter< ignite::optim >::type optimizer(optimizerSEXP);
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
// rcpp_ignite_adamw
ignite::optim_adamw rcpp_ignite_adamw(torch::TensorList params, double lr, double beta1, double beta2, double eps, double weight_decay, bool amsgrad);
RcppExport SEXP _ignite_rcpp_ignite_adamw(SEXP paramsSEXP, SEXP lrSEXP, SEXP beta1SEXP, SEXP beta2SEXP, SEXP epsSEXP, SEXP weight_decaySEXP, SEXP amsgradSEXP) {
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
    rcpp_result_gen = Rcpp::wrap(rcpp_ignite_adamw(params, lr, beta1, beta2, eps, weight_decay, amsgrad));
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
// rcpp_delete_optim
void rcpp_delete_optim(void* x);
RcppExport SEXP _ignite_rcpp_delete_optim(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_optim(x);
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
// rcpp_delete_adamw_options
void rcpp_delete_adamw_options(void* x);
RcppExport SEXP _ignite_rcpp_delete_adamw_options(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< void* >::type x(xSEXP);
    rcpp_delete_adamw_options(x);
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
    {"_ignite_rcpp_ignite_adamw_get_param_groups", (DL_FUNC) &_ignite_rcpp_ignite_adamw_get_param_groups, 1},
    {"_ignite_rcpp_ignite_optim_get_param_group_params", (DL_FUNC) &_ignite_rcpp_ignite_optim_get_param_group_params, 1},
    {"_ignite_rcpp_ignite_adamw_get_param_group_options", (DL_FUNC) &_ignite_rcpp_ignite_adamw_get_param_group_options, 1},
    {"_ignite_rcpp_ignite_adamw_set_param_group_options", (DL_FUNC) &_ignite_rcpp_ignite_adamw_set_param_group_options, 3},
    {"_ignite_rcpp_ignite_adamw_get_states", (DL_FUNC) &_ignite_rcpp_ignite_adamw_get_states, 1},
    {"_ignite_rcpp_ignite_adamw_get_state", (DL_FUNC) &_ignite_rcpp_ignite_adamw_get_state, 1},
    {"_ignite_rcpp_adamw_state_exp_avg", (DL_FUNC) &_ignite_rcpp_adamw_state_exp_avg, 1},
    {"_ignite_rcpp_adamw_state_exp_avg_sq", (DL_FUNC) &_ignite_rcpp_adamw_state_exp_avg_sq, 1},
    {"_ignite_rcpp_adamw_state_max_exp_avg_sq", (DL_FUNC) &_ignite_rcpp_adamw_state_max_exp_avg_sq, 1},
    {"_ignite_rcpp_adamw_state_step", (DL_FUNC) &_ignite_rcpp_adamw_state_step, 1},
    {"_ignite_rcpp_ignite_opt_step", (DL_FUNC) &_ignite_rcpp_ignite_opt_step, 5},
    {"_ignite_rcpp_ignite_predict_step", (DL_FUNC) &_ignite_rcpp_ignite_predict_step, 2},
    {"_ignite_rcpp_ignite_adamw", (DL_FUNC) &_ignite_rcpp_ignite_adamw, 7},
    {"_ignite_rcpp_ignite_adamw_step", (DL_FUNC) &_ignite_rcpp_ignite_adamw_step, 1},
    {"_ignite_rcpp_ignite_adamw_zero_grad", (DL_FUNC) &_ignite_rcpp_ignite_adamw_zero_grad, 1},
    {"_ignite_rcpp_delete_optim", (DL_FUNC) &_ignite_rcpp_delete_optim, 1},
    {"_ignite_rcpp_delete_optim_sgd", (DL_FUNC) &_ignite_rcpp_delete_optim_sgd, 1},
    {"_ignite_rcpp_delete_optim_adam", (DL_FUNC) &_ignite_rcpp_delete_optim_adam, 1},
    {"_ignite_rcpp_delete_optim_adamw", (DL_FUNC) &_ignite_rcpp_delete_optim_adamw, 1},
    {"_ignite_rcpp_delete_optim_adagrad", (DL_FUNC) &_ignite_rcpp_delete_optim_adagrad, 1},
    {"_ignite_rcpp_delete_optim_rmsprop", (DL_FUNC) &_ignite_rcpp_delete_optim_rmsprop, 1},
    {"_ignite_rcpp_delete_optim_param_groups", (DL_FUNC) &_ignite_rcpp_delete_optim_param_groups, 1},
    {"_ignite_rcpp_delete_optim_param_group", (DL_FUNC) &_ignite_rcpp_delete_optim_param_group, 1},
    {"_ignite_rcpp_delete_adamw_param_groups", (DL_FUNC) &_ignite_rcpp_delete_adamw_param_groups, 1},
    {"_ignite_rcpp_delete_adamw_param_group", (DL_FUNC) &_ignite_rcpp_delete_adamw_param_group, 1},
    {"_ignite_rcpp_delete_adamw_states", (DL_FUNC) &_ignite_rcpp_delete_adamw_states, 1},
    {"_ignite_rcpp_delete_adamw_state", (DL_FUNC) &_ignite_rcpp_delete_adamw_state, 1},
    {"_ignite_rcpp_delete_adamw_options", (DL_FUNC) &_ignite_rcpp_delete_adamw_options, 1},
    {"_ignite_ignite_raise_exception", (DL_FUNC) &_ignite_ignite_raise_exception, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_ignite(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
