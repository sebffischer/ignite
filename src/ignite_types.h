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

class optim_adam {
public:
  std::shared_ptr<void> ptr;
  optim_adam (void* x);
  optim_adam (std::shared_ptr<void> x) : ptr(x) {}
  optim_adam (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_adamw {
public:
  std::shared_ptr<void> ptr;
  optim_adamw (void* x);
  optim_adamw (std::shared_ptr<void> x) : ptr(x) {}
  optim_adamw (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_adagrad {
public:
  std::shared_ptr<void> ptr;
  optim_adagrad (void* x);
  optim_adagrad (std::shared_ptr<void> x) : ptr(x) {}
  optim_adagrad (SEXP x);
  operator SEXP () const;
  void* get ();
};

class optim_rmsprop {
public:
  std::shared_ptr<void> ptr;
  optim_rmsprop (void* x);
  optim_rmsprop (std::shared_ptr<void> x) : ptr(x) {}
  optim_rmsprop (SEXP x);
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
