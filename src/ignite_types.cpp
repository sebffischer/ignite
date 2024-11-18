#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"
#include <ignite/ignite.h>


Rcpp::XPtr<torch::Tensor> get_tensor_ptr(void* tensor) {
    auto xptr = make_xptr<torch::Tensor>(tensor);
    xptr.attr("class") = Rcpp::CharacterVector::create("torch_tensor", "R7");
    return xptr;
}

namespace ignite {

void* adamw_param_groups::get() {
  return ptr.get();
}

// I want to create a converter for adamw_options to SEXP


adamw_param_groups::operator SEXP () const {
  auto x = ptr.get();

  int size = rcpp_ignite_adamw_param_groups_size(x);
  Rcpp::List lst = Rcpp::List::create();
  for (int i = 0; i < size; i++) {
    Rcpp::List lst_inner = Rcpp::List::create();
    auto params = rcpp_ignite_optim_get_param_group_params(x, i);
    auto y = ignite_adamw_get_param_group_options(x, i);;
    lst_inner["params"] = params;
    lst_inner["lr"] = y.lr;
    lst_inner["weight_decay"] = y.weight_decay;
    lst_inner["betas"] = Rcpp::NumericVector::create(y.betas[0], y.betas[1]);
    lst_inner["eps"] = y.eps;
    lst_inner["amsgrad"] = y.amsgrad;

    lst.push_back(lst_inner);
  }
  return lst;
}

adamw_param_groups::adamw_param_groups(SEXP x) {
  return;
}

// we don't own the
adamw_param_groups::adamw_param_groups (void* x) : ptr(x, [](void*) {}) {};

void* optim_param_groups::get() {
  return ptr.get();
}
optim_param_groups::operator SEXP () const {
  // TODO: remoe this
  auto xptr = make_xptr<optim_param_groups>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_param_groups");
  return xptr;
}
optim_param_groups::optim_param_groups (SEXP x) : optim_param_groups{Rcpp::as<Rcpp::XPtr<optim_param_groups>>(x)->ptr} {}
optim_param_groups::optim_param_groups (void* x) : ptr(x, [](void*) {}) {};


void* optim_param_group::get() {
  return ptr.get();
}
optim_param_group::operator SEXP () const {
  auto xptr = make_xptr<optim_param_group>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_param_group");
  return xptr;
}
optim_param_group::optim_param_group (SEXP x) : optim_param_group{Rcpp::as<Rcpp::XPtr<optim_param_group>>(x)->ptr} {}
// because this is only a pointer I think we don't need to delete it.

// TODO: Is this correct? the memory of the param_group is managed by the optimizer (I think)
optim_param_group::optim_param_group (void* x) : ptr(x, [](void*) {}) {};

void* optim_options::get() {
  return ptr;
}


void* optim_adamw::get() {
  return ptr.get();
}
optim_adamw::operator SEXP () const {
  auto xptr = make_xptr<optim_adamw>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adamw");
  return xptr;
}
optim_adamw::optim_adamw (SEXP x) : optim_adamw{Rcpp::as<Rcpp::XPtr<optim_adamw>>(x)->ptr} {}
optim_adamw::optim_adamw (void* x) : ptr(x, rcpp_delete_optim_adamw) {};
}
