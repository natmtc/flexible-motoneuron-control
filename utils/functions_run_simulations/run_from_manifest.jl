# =============================================================================
# run_from_manifest.jl
#
# Drives the motoneuron pool from a scaling manifest produced by
# utils/functions_analysis_online_control_exp/unified_pipeline.py.
#
# The rule this file follows:
#
#   Julia READS already-scaled force and never rescales it.
#
# Amplitude is decided on the Python side and recorded per trial in the manifest
# (policy, baseline removal, scale factor), so a run is fully described by the
# manifest it was given plus `extra_gain`.
#
# Every output file stores its own sampling rate `fs` alongside `spike_array`,
# together with the provenance of the force that produced it.  Downstream code
# reads `fs` from the file rather than assuming a value.
#
# Usage (from the repo root):
#
#     # `using` first -- creation_model.jl needs @unpack at include time.
#     using NPZ, Interpolations, Distributions, Random, JLD2, Printf, Statistics
#     using Parameters, DSP, SymDict, Match, CSV, DataFrames
#
#     folder = "utils/functions_run_simulations/"
#     for f in sort(readdir(folder)); endswith(f, ".jl") && include(joinpath(folder, f)); end
#     MNs, nr_s = build_pool("Data_MN_Model/Data_final/Subjects_optim_process/VthCa_TA_subject1_gamma1.jld2")
#     run_manifest("Data_MN_Model/Data_final/Scaled_force/S1_global_max/manifest.csv",
#                  "Data_MN_Model/Data_final/Spikes/S1_global_max")
# =============================================================================

using NPZ, Interpolations, Distributions, Random, JLD2, Printf, Statistics
using CSV, DataFrames

# --- integration constants -------------------------------------------------
# ode4 in creation_model.jl integrates with a fixed step of hi = 0.05 ms, so
# each column of `spike_array` is 0.05 ms.  Do not change one without the other.
const INTEGRATION_STEP_MS = 0.05
const SPIKE_FS_HZ = 1000.0 / INTEGRATION_STEP_MS      # = 20000.0

# Force traces are resampled to a 1 kHz (ms) grid before integration.
const FORCE_GRID_HZ = 1000.0

# also defined in the notebooks; harmless to redefine
_int(num) = floor(Int, num)


"""
    build_pool(vthca_file; seed=1234, rheobase_jitter=0.05)

Construct the motoneuron pool and install it, plus every model constant the
ODE code reads as a global, into `Main`.

Defining the pool in one place means a parameter change applies to every run
that uses it.

`seed` fixes the rheobase jitter, so repeated runs use the same pool and results
are comparable across gains or scaling policies.

`facilitation` sets the `flag` global that `ODE_motoneuron` reads; `false` matches
the other simulation notebooks.

Returns `(MNs, nr_s)`.
"""
function build_pool(vthca_file::AbstractString; seed::Integer=1234,
                    rheobase_jitter::Real=0.05, facilitation::Bool=false)

    JLD2.@load vthca_file VthCa_newTA
    n = length(VthCa_newTA)

    rng = Random.MersenneTwister(seed)
    rheobase_S = LinRange(3.5, 6.5, n) .+ rand(rng, Normal(0.0, rheobase_jitter), n)

    rs_S  = LinRange(38.75, 41.25, n) ./ 10000
    ls_S  = LinRange(77.5, 82.5,  n) ./ 10000
    rms_S = LinRange(1.15, 1.05,  n) .* 1000
    rd_S  = LinRange(20.75, 31.25, n) ./ 10000
    ld_S  = LinRange(5500., 6800., n) ./ 10000
    rmd_S = LinRange(14.4, 10.7,  n) .* 1000

    max_gna   = 30.0
    max_gkf_S = ones(n) .* 4
    max_gks_S = LinRange(16, 25, n)
    max_gca_S = LinRange(0.038, 0.029, n)

    am   = ones(n) .* 22.0
    bm   = ones(n) .* 13.0
    ah_S = ones(n) .* 0.5
    bh_S = ones(n) .* 4.0
    an_S = ones(n) .* 1.5
    bn_S = ones(n) .* 0.1
    aq_S = ones(n) .* 1.5
    bq_S = LinRange(0.025, 0.038, n)
    ap_D = 0.008
    bp_D = LinRange(0.014, 0.016, n)

    Vthca_S   = VthCa_newTA[1, :]
    Apeak_S   = LinRange(0.103, 0.123, n)
    tpeak_S   = LinRange(110., 100., n)
    tetanic_S = LinRange(0.393, 0.491, n)

    gh = ones(n)
    IF = LinRange(1., 1., n)

    MN_list = Vector{Dict}(undef, n)
    for i in 1:n
        gls = ((2π*rs_S[i]*ls_S[i])/rms_S[i])*1e6
        gld = ((2π*rd_S[i]*ld_S[i])/rmd_S[i])*1e6
        gc  = ((2/(((70*ld_S[i])/(π*rd_S[i]^2)) + ((70*ls_S[i])/(π*rs_S[i]^2)))))*1e6
        rn  = 1 / (gls + (gld*gc)/(gld+gc))
        Vth = round(rn * rheobase_S[i], digits=2)
        cs  = 2π*ls_S[i]*rs_S[i]*1e3
        cd  = 2π*ld_S[i]*rd_S[i]*1e3
        gna = 2π*ls_S[i]*rs_S[i]*max_gna*1e3
        gkf = 2π*ls_S[i]*rs_S[i]*max_gkf_S[i]*1e3
        gks = 2π*ls_S[i]*rs_S[i]*max_gks_S[i]*1e3
        gca = 2π*ld_S[i]*rd_S[i]*max_gca_S[i]*1e3

        MN_list[i] = Dict(
            :Cs=>cs, :Cd=>cd, :Irh=>rheobase_S[i], :gc=>gc,
            :gNa=>gna, :gKs=>gks, :gKf=>gkf, :gCa=>gca,
            :gld=>gld, :gls=>gls, :αp=>ap_D, :βp=>bp_D[i],
            :αq=>aq_S[i], :βq=>bq_S[i], :αm=>am[i], :βm=>bm[i],
            :αh=>ah_S[i], :βh=>bh_S[i], :αn=>an_S[i], :βn=>bn_S[i],
            :Vth=>Vth, :VthCa=>Vthca_S[i], :Rn=>rn,
            :Tetanic=>tetanic_S[i], :Apeak=>Apeak_S[i], :IF=>IF[i],
            :tpeak=>tpeak_S[i], :γh=>gh[i], "type"=>"All Type")
    end

    # Globals the ODE code reads directly.
    global nr_s = n
    global MNs  = MN_list
    global EL = 0.0; global ENa = 120.0; global ECa = 140.0
    global EK = -10.0; global Ee = 70.0; global Ei = -16.0
    global Ge = 0.6;   global Gi = 0.5
    global Vd0 = 0.0;  global Vs0 = 0.0
    global rp = 5.0;   global pd = 1
    # Facilitation flag read by ODE_motoneuron; must exist before run_sim.
    global flag = facilitation

    println("Pool built: nr_s=$(n)  seed=$(seed)  from $(basename(vthca_file))")
    return MN_list, n
end


"""
    resample_to_ms(force, fs_in) -> (trace_ms, duration_s)

Resample a force trace onto the 1 kHz grid the integrator expects.

The input rate comes from the manifest row rather than being assumed, and the
duration is not rounded to whole seconds.
"""
function resample_to_ms(force::AbstractVector, fs_in::Real)
    n_in = length(force)
    duration_s = n_in / fs_in
    n_out = max(round(Int, duration_s * FORCE_GRID_HZ), 2)
    itp = LinearInterpolation(1:n_in, float.(collect(force)))
    return itp(range(1, n_in, length=n_out)), duration_s
end


"""
    simulate_trace(trace_ms; keep_traces=false) -> spike_array

Set up the per-run global state and integrate the whole pool over `trace_ms`
(a force trace already on the ms grid, already scaled).  `build_pool` must have
been called first.

By default only the spike raster is retained, as `Int8`.  `run_sim` additionally
stores Vs and every gating variable for the whole pool -- seven
`(nr_s x sim_time)` Float64 arrays, about 4 GB for a 21 s trial at 20 kHz with
181 neurons -- which `run_manifest` then discards.  Pass `keep_traces=true` to
allocate them and go through `run_sim` when the voltage traces are wanted.

Either path calls the same `ode4` and `spikes_from_aps` in the same order, so
the spikes are identical.
"""
function simulate_trace(trace_ms::AbstractVector; keep_traces::Bool=false)
    @assert isdefined(Main, :MNs) "call build_pool(...) before simulate_trace"

    sim_length = length(trace_ms)
    amplitude  = maximum(trace_ms)

    global tspan = 0:INTEGRATION_STEP_MS:sim_length
    sim_time = _int(sim_length / INTEGRATION_STEP_MS)

    m=0.0; h=1.0; nn=0.0; q=0.0; p=0.0
    m0=0.0; h0=1.0; n0=0.0; q0=0.0; p0=0.0
    t0=-1.0; t1=-1.0; SPIKE0=0; SPIKE1=0
    global pool_state_params = fill((m,h,nn,q,p,m0,h0,n0,q0,p0,t0,t1,SPIKE0,SPIKE1), nr_s)

    global commoninput      = trace_ms
    global inh_input        = fill(0.00001 * amplitude, sim_length)
    global full_input       = zeros((nr_s, sim_length))
    global independantnoise = zeros((nr_s, sim_length))

    if keep_traces
        global spike_arr = zeros((nr_s, sim_time))
        global vs_arr    = zeros((nr_s, sim_time))
        global pt_arr    = zeros((nr_s, sim_time))
        global mt_arr    = zeros((nr_s, sim_time))
        global ht_arr    = zeros((nr_s, sim_time))
        global nt_arr    = zeros((nr_s, sim_time))
        global qt_arr    = zeros((nr_s, sim_time))
        spike_array, _vs, _pt = run_sim()
        return spike_array
    end

    # Spikes only: same per-neuron calls as `run_sim`, without retaining the
    # voltage and gating traces.
    spikes = zeros(Int8, nr_s, sim_time)
    for i in 1:nr_s
        full_input[i, :] = commoninput
        _vd, vs, _, _pt, _mt, _ht, _nt, _qt =
            ode4(tspan, 0., 0., i, pool_state_params[i], full_input[i, :], inh_input, MNs)
        spikes[i, :] = spikes_from_aps(vs, i, MNs)
    end
    return spikes
end


"""
    run_manifest(manifest_path, out_root; days=nothing, trials=nothing,
                 extra_gain=1.0, overwrite=false)

Run every trial listed in the manifest and write one `.jld2` per trial.

Each output contains:
  `spike_array`  the raster, `(nr_s, n_samples)`, `Int8` unless `keep_traces`
  `fs`           its sampling rate in Hz -- always `SPIKE_FS_HZ` (20000),
                 set by the 0.05 ms integration step.  Read it from the file
                 rather than assuming a value.
  `force_file`, `policy`, `scale_factor`, `gain`, `subject`, `day`, `trial`
                 provenance copied from the manifest row.

`extra_gain` multiplies the already-scaled force.  It defaults to 1.0 and is
recorded in the output, so any additional gain stays visible in the results.

Set `overwrite=false` (default) to resume an interrupted sweep.
"""
function run_manifest(manifest_path::AbstractString, out_root::AbstractString;
                      days=nothing, trials=nothing, extra_gain::Real=1.0,
                      overwrite::Bool=false, keep_traces::Bool=false)

    @assert isdefined(Main, :run_sim) "run_sim() is not defined -- include the files in functions_run_simulations/ first"
    @assert isdefined(Main, :MNs) "call build_pool(...) before run_manifest"

    man = CSV.read(manifest_path, DataFrame)
    days   === nothing || (man = filter(r -> r.day   in days,   man))
    trials === nothing || (man = filter(r -> r.trial in trials, man))
    isempty(man) && error("No manifest rows selected from $manifest_path")

    mkpath(out_root)
    written = String[]

    println("Running $(nrow(man)) trials from $(basename(manifest_path))")
    println("  extra_gain = $(extra_gain)   output fs = $(SPIKE_FS_HZ) Hz")

    for r in eachrow(man)
        day_str = @sprintf("Day%02d", r.day)
        out_dir = joinpath(out_root, day_str)
        mkpath(out_dir)
        outfile = joinpath(out_dir,
            @sprintf("%s_%s_trial%02d.jld2", r.subject, day_str, r.trial))

        if !overwrite && isfile(outfile)
            println("  skip (exists): $(basename(outfile))")
            push!(written, outfile)
            continue
        end

        isfile(r.out_file) || error("Manifest points at a missing force file: $(r.out_file)\n" *
                                    "Re-run scale_forces() on the Python side.")

        force = npzread(r.out_file)
        trace_ms, duration_s = resample_to_ms(force, r.fs)
        trace_ms = trace_ms .* extra_gain

        @printf("  %s trial%02d  %.1fs  peak=%.3f ... ", day_str, r.trial,
                duration_s, maximum(trace_ms))
        flush(stdout)
        t_elapsed = @elapsed spike_array = simulate_trace(trace_ms; keep_traces=keep_traces)
        @printf("done in %.1fs (%d spikes)\n", t_elapsed, Int(sum(spike_array)))
        flush(stdout)

        fs           = SPIKE_FS_HZ
        force_file   = String(r.out_file)
        policy       = String(r.policy)
        scale_factor = Float64(r.scale_factor)
        gain         = Float64(extra_gain)
        subject      = String(r.subject)
        day          = Int(r.day)
        trial        = Int(r.trial)

        JLD2.@save outfile spike_array fs force_file policy scale_factor gain subject day trial
        push!(written, outfile)
    end

    println("Wrote $(length(written)) files under $out_root")
    return written
end
