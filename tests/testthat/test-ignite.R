test_that("multiplication works", {
  library(torch)
  library(ignite)
  n = nn_linear(1, 1)
  input = torch_randn(1)
  target = torch_randn(1)

  nf = jit_trace(nn_linear(1, 1), input)
  loss_fn = jit_trace(nn_mse_loss(), input, input)

  o = ignite:::rcpp_ignite_sgd(nf$parameters, lr = 0.01, momentum = 0, dampening = 0, weight_decay = 0, nesterov = FALSE)

  nf$parameters[[1]]

  result = ignite:::rcpp_ignite_opt_step(
    # this is the correct external pointer
    network = mlr3misc::get_private(attr(nf, "module"))$ptr,
    loss_fn = mlr3misc::get_private(attr(loss_fn, "module"))$ptr,
    input   = list(input),
    target  = target,
    optimizer = o
  )
  print(loss)

  nf$parameters[[1]]
})

test_that("raise exceptions", {

  expect_error(ignite_raise_exception(), "Error from IGNITE")

})
