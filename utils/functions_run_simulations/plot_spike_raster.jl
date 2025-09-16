using CairoMakie

function plot_spike_raster(spike_array::Array{Float64,2}; highlight_neurons=Int[])
    n_neurons, n_times = size(spike_array)
    CairoMakie.activate!()

    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1, 1], title="Raster Plot", xlabel="Time (samples)", ylabel="Neuron Index")
    line_height = 0.5
    line_thickness = 0.5
    horizontal_offset = 100  # spacing between neurons

    for t in 1:n_times
        for neuron in 1:n_neurons
            if spike_array[neuron, t] == 1.0
                x_offset = horizontal_offset * (neuron - 1)
                x = t + x_offset
                p1 = CairoMakie.Point2f(x, neuron - line_height / 2)
                p2 = CairoMakie.Point2f(x, neuron + line_height / 2)

                color = neuron in highlight_neurons ? :red : :black
                CairoMakie.lines!(ax, [p1, p2], color=color, linewidth=line_thickness)
            end
        end
    end

    ax.xgridvisible = false
    ax.ygridvisible = false
    return fig
end
