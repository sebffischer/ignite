#pragma once

extern "C" struct adamw_options {
    double lr;
    double weight_decay;
    double betas[2];
    double eps;
    bool amsgrad;

    adamw_options() = default;


    adamw_options(long x) { }
};

