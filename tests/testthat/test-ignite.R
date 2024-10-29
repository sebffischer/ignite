test_that("Igniter", {
  library(ignite)
  library(torch)
  n = nn_linear(1, 1)
  input = torch_randn(1)
  target = torch_randn(1)

  nf = jit_trace(nn_linear(1, 1), input)
  nf$to(device = "mps")
  loss_fn = jit_trace(nn_mse_loss(), input, input)

  opt = optim_ignite_adam(nf$parameters)

  igniter = Igniter$new(
    network = nf,
    loss_fn = loss_fn,
    target = target,
    optimizer = opt
  )

  igniter$opt_step(
    list(torch_randn(10, 1)$to(device = "mps"))
  )


  nf$parameters[[1]]
})

test_that("raise exceptions", {

  expect_error(ignite_raise_exception(), "Error from IGNITE")

})
