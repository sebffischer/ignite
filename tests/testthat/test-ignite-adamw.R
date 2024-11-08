make_adamw = function(...) {
  n = torch::nn_linear(1, 1)

  o = optim_ignite_adamw(n$parameters, ...)
  x = torch_randn(10, 1)
  y = torch_randn(10, 1)
  loss = mean((n(x) - y)^2)
  loss$backward()
  o$step()
  o
}

test_that("constructor arguments are passed to the optimizer", {
  # TODO
})

test_that("param_groups works", {
  # TODO: Write a lot of tests
  library(ignite)
  library(torch)

  o = make_adamw(lr = 0.1)
  o$state_dict2()
  o$param_groups

  ignite:::rcpp_ignite_adamw_state(o$ptr)
  # TODO: Something is off with the parameter conversion


  o$param_groups[[1]]$lr = 100
  expect_equal(o$param_groups[[1]]$lr, 100)
})

test_that("state_dict works", {
  o = make_sgd(lr = 0.1, momentum = 0.9)
  sd = o$state_dict()
  expect_equal(length(sd), 1)
  expect_equal(sd[[1]]$lr, 0.1)
})
