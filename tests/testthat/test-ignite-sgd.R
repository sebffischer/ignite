make_sgd = function(...) {
  n = torch::nn_linear(1, 1)
  x = torch_randn(10, 1)
  y = torch_randn(10, 1)
  optim_ignite_sgd(n$parameters, ...)
}

test_that("constructor arguments are passed to the optimizer", {
  # TODO
})

test_that("param_groups works", {
  # TODO: Write a lot of tests
  library(ignite)
  library(torch)

  o = make_sgd(lr = 0.1, momentum = 0.9)

  o$param_groups[[1]]$lr = 100
  expect_equal(o$param_groups[[1]]$lr, 100)
})

test_that("state_dict works", {
  o = make_sgd(lr = 0.1, momentum = 0.9)
  sd = o$state_dict()
  expect_equal(length(sd), 1)
  expect_equal(sd[[1]]$lr, 0.1)
})

test_that("...", {
  library(ignite)
  library(torch)
  n = torch::nn_linear(1, 1)
  o = optim_ignite_sgd(n$parameters, lr = 0.1, momentum = 0.9)
  o = optim_ignite_adamw(n$parameters, lr = 0.1)
  n$parameters[[1]]
  x = torch_randn(10, 1)
  y = torch_randn(10, 1)
  loss = mean((n(x) - y)^2)
  loss$backward()
  o$step()
  n$parameters[[1]]
})


test_that("sgd can be cleaned up", {
  # this used to segfault.
  o = optim_ignite_sgd(torch::nn_linear(1, 1)$parameters, lr = 0.1)
  rm("o")
  gc()
  gc()
  expect_true(TRUE)
})
