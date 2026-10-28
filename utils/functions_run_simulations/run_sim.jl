"""
    run_sim()

Simulate the whole pool of `nr_s` motoneurons driven by a common excitatory input.

Loops over every neuron, integrates it with `ode4`, extracts spikes with `spikes_from_aps`,
and stores the results into the pre-allocated global arrays. Reads the globals
`nr_s`, `commoninput`, `tspan`, `pool_state_params`, `inh_input`, `MNs`, and writes the
storage arrays (`spike_arr`, `vs_arr`, `pt_arr`, `mt_arr`, `ht_arr`, `nt_arr`, `qt_arr`).

# Returns
`(spike_arr, vs_arr, pt_arr)` — the pool spike trains, somatic voltages, and Ca-gate traces.
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