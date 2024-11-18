#include <Rcpp.h>
#include "ignite_types.h"
#include "exports.h"
#include <ignite/exports.h>

// [[Rcpp::export]]
void rcpp_ignite_adamw_set_param_group_options(ignite::optim_adamw opt, Rcpp::List list) {
  for (int i = 0; i < list.length(); i++) {
    Rcpp::List group_options = list[i];
    adamw_options opts;
    opts.lr = group_options["lr"];

    Rcpp::NumericVector betas = group_options["betas"];
    opts.betas[0] = betas[0];
    opts.betas[1] = betas[1];
    opts.eps = group_options["eps"];
    opts.weight_decay = group_options["weight_decay"];
    opts.amsgrad = group_options["amsgrad"];
    ignite_adamw_set_param_group_options(opt.get(), i, opts);
  }
}
