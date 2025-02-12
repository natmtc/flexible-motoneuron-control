"""
run_sim, runs the simulation for the #nr_s motoneurons considering a common input to all the motoneurons
"""
function run_sim()

    for i in 1:nr_s
        full_input[i,:] = commoninput 
        vd, vs, _,pt, mt, ht, nt, qt = ode4(tspan, 0.,0., i, pool_state_params[i], full_input[i,:], inh_input, MNs)      
        spikes = spikes_from_aps(vs, i, MNs)       
        spike_arr[i,:] = spikes
        vs_arr[i,:] = vs
        pt_arr[i,:] = pt;mt_arr[i,:] = mt;ht_arr[i,:] = ht;nt_arr[i,:] = nt;qt_arr[i,:] = qt
    end
    return spike_arr, vs_arr,pt_arr
end