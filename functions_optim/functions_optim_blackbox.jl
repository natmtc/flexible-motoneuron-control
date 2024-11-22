function object_function_constant_inh(VthCa,i)
    simulated_hysteresis, _ = simulate_hh_hysteresis_constant_inh(VthCa,i,MN_list)
    ssd = sum((hysteresis_exp[i] .- simulated_hysteresis).^2)
    return ssd 
end


function run_sim_optim_constant_inh(i)
    full_input = commoninput 
    _, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input, inh_input, MNs)       
    spikes = spikes_from_aps(vs, i, MNs)       
    spike_arr = reshape(spikes, 1, length(spikes))
    vs_arr[i,:] = vs
    return spike_arr, vs_arr
end


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
