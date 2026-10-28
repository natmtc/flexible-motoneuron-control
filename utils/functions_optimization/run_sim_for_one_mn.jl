"""
    run_sim_inh(i, inh_input)

Simulate a single motoneuron (index `i`) with the given inhibitory input, using the
no-facilitation model (`ode4` from `creation_model_nofacilitation.jl`).

Reads the globals `commoninput`, `tspan`, `pool_state_params`, `MNs`, `vs_arr`.

# Returns
- `spike_arr` : 1×T binary spike train for neuron `i`.
- `vs_arr`    : the pool voltage array with row `i` filled in.
"""
function run_sim_inh(i,inh_input)
    full_input = commoninput
    vd, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input,inh_input, MNs)  
    spikes = spikes_from_aps(vs, i, MNs)       
    spike_arr = reshape(spikes, 1, length(spikes))
    vs_arr[i,:] = vs
    return spike_arr, vs_arr
end