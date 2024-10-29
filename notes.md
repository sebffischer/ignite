# Architecture

## Findings:

* In order to pass a `list()` as `torch::jit::Stack` I need to implement the conversion
  from `list()` <-> `Stack`: https://github.com/mlverse/torch/blob/ab71bda8de0507770b04835c5c4838e91f3c6261/src/torch_api.cpp#L1007-L1024
  Also see: https://github.com/mlverse/torch/blob/ab71bda8de0507770b04835c5c4838e91f3c6261/inst/include/torch_impl.h#L286-L288

* In order to pass a `script_method` as a `torch::jit::Function` (and then cast it to `GraphFunction`), I need to implement:
https://github.com/mlverse/torch/blob/ab71bda8de0507770b04835c5c4838e91f3c6261/inst/include/torch_types.h#L318


## (trace) jitting

* `torch::jit::Stack` is an R `list()` I think (see `make_traceable_fn` comment)
* When doing `jit_trace(nnf_relu, torch_randn(1))` we get a `ScriptFunction`
  that contains as a pointer the result of `cpp_trace_function` (R function)
  This function then calls via `.Call` into `_torch_cpp_trace_function`
  This function casts:
  * the first argument to an `Rcpp::Function`
  * the second argument as a `list(...)` into a `XPtrTorchStack`
  * ... (not interesting)

    This function then calls into:
    `cpp_trace_function` (the actual rcpp function):
      Here I think the actual jitting is done by calling `.create_function()` from
      `CompilationUnit` which is then wrapped in a `XPtrTorchFunctionPtr`
      this `.create_function()` which returns returns a `torch::jit::Function*`.
      (`GraphFunction` is derived from `Function`). This works because C++ can
      automatically cast derived classes to their derivee.

      This `Function*` is then wrapped in a `XPtrTorchFunctionPtr` (including its deleter).
      This class is derived from `XPtrTorch` and contains a `std::shared_ptr<void*>`
      This `XPtrTorchFunctionPtr` is then converted to an external pointer using `make_xptr`.
      This allowcates `XPtrTorchFunctionPtr` on the heap and then wraps this in an external pointer.

    then, the `_torch_cpp_trace_function` converts the `Rcpp::XPtr` (using `Rcpp::wrap()`) which results
    in an external pointer `SEXP`.

  The `jit_trace` function then wraps this `ptr` in `ScriptFunction`.

  For the forward pass, `cpp_call_traced_fn` is called, which takes in:
  * `fn`: the external pointer created above:
  * `list()` of tensors.

  In `_torch_cpp_call_traced_fn`:
  * `fn` is converted to `XPtrTorchFunctionPtr` <-- This is important
  * `list()` is converted to `XPtrTorchStack` <-- This is also important

  To call into `torch`, these now need to be converted to `void*` to call into the lantern API:
  For both we get the `void*` by calling the `get()` method.

  Then we call into `lantern_call_traced_fn` which does some error handling and mostly calls
  into `_lantern_call_traced_fn`

  ```r
  auto fn_ = reinterpret_cast<GraphFunction*>(fn);
  Stack inputs_ = *reinterpret_cast<Stack*>(inputs);

  auto outputs = torch::jit::Stack();
  auto out = (*fn_)(inputs_);
  outputs.push_back(out);

  return make_ptr<torch::jit::Stack>(outputs);
  ```


How I currently implement it:

* There is a `rcpp_ignite_run_script_module()` function in R which calls
  `_ignite_rcpp_ignite_run_script_module`.
  This casts the `SEXP` to a `ignite::graph_function` and calls `rcpp_ignite_script_module`
  which calls `ignite_run_script_module()` on what is returned by `.get()`, which gets a `void*`
  from the `shared_ptr<>`

