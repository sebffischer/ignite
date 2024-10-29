#pragma once
#include <torch.h>

namespace ignite {

class optim_sgd {
public:
  std::shared_ptr<void> ptr;
  optim_sgd (void* x);
  optim_sgd (std::shared_ptr<void> x) : ptr(x) {}
  optim_sgd (SEXP x);
  operator SEXP () const;
  void* get ();
};

class stack {
public:
  std::shared_ptr<void> ptr;
  // constructor from a void* pointer;
  stack (void* x);
  // constructor from a shared_ptr<void> pointer;
  stack (std::shared_ptr<void> x) : ptr(x) {}
  // constructor from an R object;
  stack (SEXP x);
  // implicit casting operator
  operator SEXP () const;
  // conversion to a void* pointer;
  void* get ();
};

class graph_function {
public:
  std::shared_ptr<void> ptr;
  // constructor from a void* pointer;
  graph_function (void* x);
  // constructor from a shared_ptr<void> pointer;
  graph_function (std::shared_ptr<void> x) : ptr(x) {}
  // constructor from an R object;
  graph_function (SEXP x);
  // implicit casting operator
  operator SEXP () const;
  // conversion to a void* pointer;
  void* get ();
};
}
