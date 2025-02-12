function objective_function(temporary_VCa, i, MN_list, hysteresis_exp)
    
    total_error = 0

    hysteresis, _ = simulate_hh_hysteresis_constant_inh(temporary_VCa, i, MN_list)

    if hysteresis != 100  # 100 is a flag for no spikes
        error = (hysteresis - hysteresis_exp[i])^2
        total_error += error
    else
        total_error += 1.0
    end

    return total_error
end

function wrap_objective(x, i, Vthca_S, MN_list, hysteresis_exp) 

    temporary_VCa = copy(Vthca_S) 

    temporary_VCa[i] = x[1] #x is the candidate solution for VthCa for a given neuron

    ssd = objective_function(temporary_VCa, i, MN_list, hysteresis_exp)
    
    return ssd #total error, sum of squared distances
end
