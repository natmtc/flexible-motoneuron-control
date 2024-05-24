function run_sim_optim_constant_inh()
    for i in 1:nr_s
        # temp = filter_input(50,rand(Normal(0, 1),sim_length+2000))  # add some gaussian noise - optional
        # independantnoise[i,:] = temp[1001:end-1000] * 0.25;
        full_input[i,:] = commoninput  + independantnoise[i,:]; 
        vd, vs, _,pt, mt, ht, nt, qt = ode4(tspan, 0.,0., i, pool_state_params[i], full_input[i,:], inh_input, MNs)      
        spikes = spikes_from_aps(vs, i, MNs)       
        spike_arr[i,:] = spikes
        vs_arr[i,:] = vs
        pt_arr[i,:] = pt;mt_arr[i,:] = mt;ht_arr[i,:] = ht;nt_arr[i,:] = nt;qt_arr[i,:] = qt
    end
    return spike_arr, vs_arr,pt_arr
end