"""
run_sim
Runs the simulation for the #nr_s motoneurons considering a common input to all the motoneurons
"""

function run_sim()
    for i in 1:nr_s

        temp = filter_input(50,rand(Normal(0, 1),sim_length+2000))  # add some gaussian noise - optional
        independantnoise[i,:] = temp[1001:end-1000] * 0.25;

        full_input[i,:] = commoninput + independantnoise[i,:]; 
        
        vd, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input[i,:], full_input[i,:]*factor, MNs)      
        spikes = spikes_from_aps(vs, i, MNs)     
        spike_arr[i,:] = spikes
        vs_arr[i,:] = vs
    end
    return spike_arr, vs_arr
end