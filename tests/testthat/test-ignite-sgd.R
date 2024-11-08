make_sgd = function(...) {
  n = torch::nn_linear(1, 1)
  x = torch_randn(10, 1)
  y = torch_randn(10, 1)
  optim_sgd(n$parameters, ...)
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
