#############################################################
#############################################################
############ FUNCTIONS FOR OPTIMISATION #####################
#############################################################
#############################################################


function object_function(γh,i)
    penalty = 0
    penalty_factor = 1e6  # Adjust

    if γh[i] < 0. || γh[i] > 1.
        penalty += penalty_factor * (0 - γh[i])^2
    end

    simulated_hysteresis, _ = simulate_hh_hysteresis(γh,i,MN_list)
    ssd = sum((hysteresis_exp[i] .- simulated_hysteresis).^2)
    
    return ssd + penalty
end


function run_sim_optim(i)

    full_input = commoninput 
    _, vs, _ = ode4(tspan, 0.,0., i, pool_state_params[i], full_input, full_input*factor, MNs)       
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


function simulate_hh_hysteresis(γh::Vector,i,MN_list)    
    MN_list[i][:γh] = γh[i]

    MNs= MN_list[i]

    spike_array, _ = run_sim_optim(i);

    amplitude=0.4;
    sim_length=560000 ;
    half_triang = div(sim_length, 2);  
    flat_line = fill(-0.1, 20000);
    up = range(-0.1, amplitude, length = half_triang + 1)  ;
    down = range(amplitude, -0.1, length = sim_length - half_triang + 1);  
    tr = vcat(flat_line, up[2:end], down[2:end], flat_line);
    equiv_force_profile= (tr/maximum(tr))*nr_s

    hysteresis= []

    value_1= equiv_force_profile[findfirst(==(1),spike_array[1,:]),:] #force at the recruitment threshold
    value_2=  equiv_force_profile[findlast(==(1),spike_array[1,:]),:] #force at the derecruitment threshold

    if value_1 !== nothing
        # Safe to use index
    else
        print(11111111)
    end

    if value_2 !== nothing
        # Safe to use index
    else
        print(222222222)
    end
    hysteresis= (value_2/value_1).-1
    # println(hysteresis)

    max_firing_rates= firing_rates_max(spike_array,i)
    
    return hysteresis, max_firing_rates
end


function compute_gradient_manual!(g::Vector{Float64}, γh::Vector{Float64}, i::Int; ε::Float64=1e-3)
    γh_plus = copy(γh)
    γh_minus = copy(γh)
    
    γh_plus[i] += ε
    γh_minus[i] -= ε

    f_plus = BigFloat(object_function(γh_plus, i))
    f_minus = BigFloat(object_function(γh_minus, i))

    g[i] = (f_plus - f_minus) / (2 * ε)
end

function gradient_descent_optimization(γh_initial::Vector{Float64}, i::Int, learning_rate::Float64, max_iterations::Int, tolerance::Float64)
    
    γh = copy(γh_initial)  # Initial parameters
    g = zeros(Float64, length(γh))  # Gradient vector initialization
    obj_value_prev = BigFloat(object_function(γh, i))  # Initial objective function value

    for iter in 1:max_iterations
        fill!(g, 0.0)  # Reset gradient vector
        compute_gradient_manual!(g, γh, i)  # Compute gradient

        
        for j in 1:length(γh)
            γh[j] -= learning_rate * g[j]
        end

        # Ensure γh remains within bounds [0, 1]
        γh = clamp.(γh, 0.0, 1.0)

        # Compute new objective function value
        obj_value_new = BigFloat(object_function(γh, i))


        if abs(obj_value_new - obj_value_prev) < tolerance
            println("Convergence achieved at iteration $iter")
            break  
        end

        obj_value_prev = obj_value_new  # Update previous objective function value for next iteration

        println("Iteration $iter, Objective Function Value: ", obj_value_new)
    end

    return γh  
end

