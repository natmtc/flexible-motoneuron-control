"""
    object_function_constant_inh(VthCa, i)

Compute the sum of squared deviations (SSD) between experimental hysteresis for index `i`
and the simulated hysteresis produced with calcium threshold `VthCa` under constant inhibition.

# Inputs
- `VthCa`: calcium spike threshold parameter for the simulation.
- `i`: index of the dataset/condition to compare.

# Outputs
- `ssd`: scalar sum of squared differences between `hysteresis_exp[i]` and the simulation.
"""
function object_function_constant_inh(VthCa,i)
    simulated_hysteresis, _ = simulate_hh_hysteresis_constant_inh(VthCa,i,MN_list)
    ssd = sum((hysteresis_exp[i] .- simulated_hysteresis).^2)
    return ssd 
end




"""
    run_sim_optim_constant_inh(i)

Run the constant-inhibition simulation for dataset/index `i`, compute spikes from somatic voltage,
and store the voltage trace.

# Inputs
- `i`: index of the motoneuron/dataset to simulate.

# Outputs
- `spike_arr`: 1×T array of detected spikes.
- `vs_arr`: matrix with the updated somatic voltage trace stored at row `i`.
"""
function run_sim_optim_constant_inh(i)
    full_input = commoninput 
    _, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input, inh_input, MNs)       
    spikes = spikes_from_aps(vs, i, MNs)       
    spike_arr = reshape(spikes, 1, length(spikes))
    vs_arr[i,:] = vs
    return spike_arr, vs_arr
end


"""
    firing_rates_max(spike_array, i)

Compute the maximum instantaneous firing rate (Hz) per motoneuron from a binary spike matrix.

# Inputs
- `spike_array`: matrix (neurons × time) with 1.0 indicating a spike.
- `i`: unused index parameter (kept for interface consistency).

# Outputs
- `max_rate_mn`: maximum firing rate (Hz) as computed by the function.
"""
function firing_rates_max(spike_array,i)
    max_rates_mns=[]
    for i in 1: size(spike_array)[1]
        spike_pos=[]
        for j in 1:size(spike_array)[2]
            if spike_array[i,j]==1.0
                push!(spike_pos,j)
            end
        end
        ISI= diff(spike_pos);
        rates= 1 ./(ISI/20000)
        max_rates= maximum(rates)
        max_rates_mns=max_rates    
    end 
    max_rate_mn= max_rates_mns;
    return max_rate_mn;
    end


"""
    simulate_hh_hysteresis_constant_inh(values_deac, i, MN_list)

Run a constant-inhibition HH simulation for motoneuron `i`, set its Ca-threshold from `values_deac`,
and compute hysteresis as (derecruitment/recruitment − 1). Returns hysteresis and a placeholder for
max firing rates.

# Inputs
- `values_deac`: vector of Ca-threshold values; uses `values_deac[i]` for MN `i`.
- `i`: index of the motoneuron to simulate.
- `MN_list`: list/array of motoneuron parameter dictionaries/structs.

# Outputs
- `hysteresis`: scalar hysteresis measure (Float) or 100 if no spikes detected.
- `max_firing_rates`: (currently empty) vector placeholder for max firing rates.
"""
function simulate_hh_hysteresis_constant_inh(values_deac::Vector,i,MN_list) 
      
    MN_list[i][:VthCa] = values_deac[i]
    MNs= MN_list[i]
    spike_array, _ = run_sim_optim_constant_inh(i);

    amplitude=0.4;
    sim_length=560000 ;
    half_triang = div(sim_length, 2);  
    flat_line = fill(-0.1, 20000);
    up = range(-0.1, amplitude, length = half_triang + 1)  ;
    down = range(amplitude, -0.1, length = sim_length - half_triang + 1);  
    tr = vcat(flat_line, up[2:end], down[2:end], flat_line);
    hysteresis= []

    if any(spike_array[1, :] .== 1)     
        value_1= tr[findfirst(==(1),spike_array[1,:]),:] #force at the recruitment threshold
        value_2=  tr[findlast(==(1),spike_array[1,:]),:] #force at the derecruitment threshold
        hysteresis= (value_2[1]/value_1[1]).- 1
    else
        hysteresis=100 #random high value
    end

    max_firing_rates= [] #not relevant now firing_rates_max(spike_array,i) SOLVE THE CASE IN WHICH THERE ARE NO SPIKES
    
    return hysteresis, max_firing_rates
end


"""
    objective_function(temporary_VCa, i, MN_list, hysteresis_exp)

Evaluate the SSD-style error between simulated hysteresis (with candidate VCa) and the experimental
hysteresis for index `i`, penalizing trials with no spikes.

# Inputs
- `temporary_VCa`: vector of candidate Ca-threshold values.
- `i`: index of the motoneuron/dataset to evaluate.
- `MN_list`: list of motoneuron parameter sets.
- `hysteresis_exp`: vector of experimental hysteresis values.

# Outputs
- `total_error`: scalar error (Float64).
"""
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



"""
    wrap_objective(x, i, Vthca_S, MN_list, hysteresis_exp)

Replace `Vthca_S[i]` with candidate `x[1]`, run the objective, and return its SSD error.

# Inputs
- `x`: 1-element vector with candidate VthCa for neuron `i`.
- `i`: index of the neuron/dataset.
- `Vthca_S`: vector of baseline VthCa values.
- `MN_list`: motoneuron parameter sets.
- `hysteresis_exp`: experimental hysteresis values.

# Outputs
- `ssd`: scalar sum-of-squared error for the candidate.
"""
function wrap_objective(x, i, Vthca_S, MN_list, hysteresis_exp) 

    temporary_VCa = copy(Vthca_S) 

    temporary_VCa[i] = x[1] #x is the candidate solution for VthCa for a given neuron

    ssd = objective_function(temporary_VCa, i, MN_list, hysteresis_exp)
    
    return ssd #total error, sum of squared distances
end
