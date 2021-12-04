function y = activation_tanh(alpha) 

    y = (exp(alpha) - exp(-alpha)) ./ (exp(alpha) + exp(-alpha));
end