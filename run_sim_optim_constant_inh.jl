function run_sim_optim_constant_inh()
    for i in 1:1

        full_input[i,:] = commoninput # + independantnoise[i,:]; 
        vd, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input[i,:], inh_input, MNs)      
        spikes = spikes_from_aps(vs, i, MNs)       
        spike_arr = spikes
        vs_arr[i,:] = vs
    end
    return spike_arr, vs_arr
end