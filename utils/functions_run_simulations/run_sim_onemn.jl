"""
    run_sim_onemn(MN_list, i, inh_input)

Simulate a **single** motoneuron (index `i`) with the given inhibitory input.

Uses the globals `commoninput` (common excitatory drive), `tspan`, `pool_state_params`,
`MNs`, and `vs_arr`. Integrates with `ode4`, then extracts spikes with `spikes_from_aps`.

# Returns
- `spike_arr` : 1×T binary spike train for neuron `i`.
- `vs_arr`    : the pool voltage array with row `i` filled in.
"""
function run_sim_onemn(MN_list,i,inh_input) #tuned for one MN
    full_input = commoninput
    _, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input,inh_input, MNs)       
    spikes = spikes_from_aps(vs, i, MNs)       
    spike_arr = reshape(spikes, 1, length(spikes))
    vs_arr[i,:] = vs
    return spike_arr, vs_arr
end