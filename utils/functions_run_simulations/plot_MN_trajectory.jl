"""
    plot_MN_trajectory(spike_array, idx1, idx2; v=100)

Compute and plot a 2D firing-rate trajectory between two motoneurons by smoothing their spike trains,
block-averaging over windows of size `v`, and drawing time-colored line segments.

"""
function plot_MN_trajectory(spike_array::Matrix{Float64}, idx1::Int, idx2::Int; v::Int = 100) #v is window size to avg


    smoothed1, smoothed2 = conv_spikes_rates(reshape(spike_array[idx1, :], 1, :),reshape(spike_array[idx2, :], 1, :))

    trajectory = hcat(smoothed1, smoothed2)

    # Block-averaging
    n = div(size(trajectory, 1), v)
    avg_traj = zeros(n, 2)
    for i in 1:n
        start_idx = (i - 1) * v + 1
        end_idx = i * v
        avg_traj[i, 1] = mean(trajectory[start_idx:end_idx, 1])
        avg_traj[i, 2] = mean(trajectory[start_idx:end_idx, 2])
    end

    # Color map for time progression
    color_map = [get(ColorSchemes.viridis, i) for i in LinRange(0, 1, size(avg_traj, 1))]

    # Build segments
    lines_x = []
    lines_y = []
    for i in 1:(size(avg_traj, 1) - 1)
        push!(lines_x, [avg_traj[i, 1], avg_traj[i + 1, 1]])
        push!(lines_y, [avg_traj[i, 2], avg_traj[i + 1, 2]])
    end

    # Plot
    p = Plots.plot(legend=false, grid=false, xlabel="MN $(idx1) firing rate", ylabel="MN $(idx2) firing rate", title="Firing Rate Trajectory")
    for i in 1:length(lines_x)
        Plots.plot!(p, lines_x[i], lines_y[i], color=color_map[i], linewidth=4)
    end

    return p, avg_traj
    
end
