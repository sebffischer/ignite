test_that("multiplication works", {
  n = nn_linear(1, 1)
  input = torch_randn(1)
  target = torch_randn(1)

  nf = jit_trace(nn_linear(1, 1), input)
  loss_fn = jit_trace(nnf_mse_loss, input, input)

  o = ignite:::rcpp_ignite_sgd(nf$parameters, lr = 0.01, momentum = 0, dampening = 0, weight_decay = 0, nesterov = FALSE)

  nf$parameters[[1]]

  loss = ignite:::rcpp_ignite_run_script_module(
    # this is the correct external pointer
    network = mlr3misc::get_private(attr(nf, "module"))$ptr,
    loss_fn = loss_fn$ptr,
    input   = input,
    target  = target,
    optimizer = o
  )

  nf$parameters[[1]]
})

test_that("raise exceptions", {

  expect_error(ignite_raise_exception(), "Error from IGNITE")

})
