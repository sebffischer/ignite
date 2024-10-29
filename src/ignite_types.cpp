#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"

namespace ignite {

// optim_sgd implementation

void* optim_sgd::get() {
  return ptr.get();
}

optim_sgd::operator SEXP () const {
  auto xptr = make_xptr<optim_sgd>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("ignite_optim_sgd");
  return xptr;
}

optim_sgd::optim_sgd (SEXP x) : optim_sgd{Rcpp::as<Rcpp::XPtr<optim_sgd>>(x)->ptr} {}

optim_sgd::optim_sgd (void* x) : ptr(x, rcpp_delete_optim_sgd) {};


void* graph_function::get() {
  return ptr.get();
}

// graph_function implementation

graph_function::operator SEXP () const {
  auto xptr = make_xptr<graph_function>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("ignite_graph_function");
  return xptr;
}

graph_function::graph_function (SEXP x) : graph_function{Rcpp::as<Rcpp::XPtr<graph_function>>(x)->ptr} {}

graph_function::graph_function (void* x) : ptr(x, rcpp_delete_graph_function) {};

// stack implementation

void* stack::get() {
  return ptr.get();
}

stack::operator SEXP () const {
  auto xptr = make_xptr<stack>(*this);
  xptr.attr("class") = Rcpp::CharacterVector::create("ignite_stack");
  return xptr;
}

stack::stack (SEXP x) : stack{Rcpp::as<Rcpp::XPtr<stack>>(x)->ptr} {}

stack::stack (void* x) : ptr(x, rcpp_delete_stack2) {};
}
