#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"

namespace ignite {

void* optim_sgd::get() {
  return ptr.get();
}
optim_sgd::operator SEXP () const {
  auto xptr = make_xptr<optim_sgd>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_sgd");
  return xptr;
}
optim_sgd::optim_sgd (SEXP x) : optim_sgd{Rcpp::as<Rcpp::XPtr<optim_sgd>>(x)->ptr} {}
optim_sgd::optim_sgd (void* x) : ptr(x, rcpp_delete_optim_sgd) {};

void* optim_adam::get() {
  return ptr.get();
}
optim_adam::operator SEXP () const {
  auto xptr = make_xptr<optim_adam>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adam");
  return xptr;
}
optim_adam::optim_adam (SEXP x) : optim_adam{Rcpp::as<Rcpp::XPtr<optim_adam>>(x)->ptr} {}
optim_adam::optim_adam (void* x) : ptr(x, rcpp_delete_optim_adam) {};

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

void* optim_adagrad::get() {
  return ptr.get();
}
optim_adagrad::operator SEXP () const {
  auto xptr = make_xptr<optim_adagrad>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_adagrad");
  return xptr;
}
optim_adagrad::optim_adagrad (SEXP x) : optim_adagrad{Rcpp::as<Rcpp::XPtr<optim_adagrad>>(x)->ptr} {}
optim_adagrad::optim_adagrad (void* x) : ptr(x, rcpp_delete_optim_adagrad) {};

void* optim_rmsprop::get() {
  return ptr.get();
}
optim_rmsprop::operator SEXP () const {
  auto xptr = make_xptr<optim_rmsprop>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("optim_ignite_rmsprop");
  return xptr;
}
optim_rmsprop::optim_rmsprop (SEXP x) : optim_rmsprop{Rcpp::as<Rcpp::XPtr<optim_rmsprop>>(x)->ptr} {}
optim_rmsprop::optim_rmsprop (void* x) : ptr(x, rcpp_delete_optim_rmsprop) {};
}
