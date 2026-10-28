"""
    ODE_motoneuron(t, vd, vs, gc, gld, gls, gCa, Cd, Cs, αp, βp, VthCa, gNa, αm, βm, Vth,
                   gKf, gKs, αh, βh, αn, βn, αq, βq, γh, IF, m, h, n, q, p,
                   m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, exc_input, inh_input, flag, last_spike)

One evaluation of the two-compartment motoneuron ODE (called inside the RK4 solver `ode4`).

Given the current dendrite/soma voltages (`vd`, `vs`) and gating state, it:
- updates the somatic action-potential logic and the Na/K gating variables `m, h, n, q`
  (opening while a spike is active, closing otherwise);
- updates the dendritic calcium gate `p`, whose activation above `VthCa` drives the
  persistent inward current (PIC) responsible for recruitment hysteresis;
- returns the voltage derivatives for the soma (Na, Kf, Ks, leak, coupling) and the
  dendrite (Ca/PIC, leak, excitatory + inhibitory synapse, coupling).

# Returns
`(new_vd, new_vs, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1)` — the two
voltage derivatives followed by the updated gating variables and spike-timing state.
"""
function ODE_motoneuron(t::Float64, vd::Float64, vs::Float64,gc::Float64, gld::Float64, gls::Float64, gCa::Float64, Cd::Float64,Cs::Float64,  αp::Float64,  βp::Float64, VthCa::Float64, gNa::Float64,
    αm::Float64, βm::Float64, Vth::Float64, gKf::Float64, gKs::Float64, αh::Float64, βh::Float64,αn::Float64, βn::Float64, αq::Float64, βq::Float64, γh::Float64,IF::Float64, m::Float64, h::Float64, 
    n::Float64, q::Float64, p::Float64, m0::Float64, h0::Float64, n0::Float64, q0::Float64, p0::Float64,t0::Float64, t1::Float64, SPIKE0::Int64, SPIKE1::Int64, exc_input,inh_input,flag,last_spike)
    
    
    ### AP 
    if vs > Vth && (t0 == -1 || (t-t0) > rp)   # initiation of the AP when AP is greater than Vth
        SPIKE0 = 1; t0=t; m0=m; h0=h; n0=n;q0=q;#p0=p
    
    elseif SPIKE0 == 1 && (t-t0) > pd    # end of the AP when pulse duration ends
        SPIKE0 = 0; t0=t; m0=m; h0=h; n0=n; q0=q;#p0=p
    
    end
    
    if SPIKE0 > 0 #AP (forward equations, opening of the channels)
    
        @fastmath m = 1+(m0-1) * exp(-αm*(t-t0)) 
        @fastmath h = h0*exp(-βh*(t-t0)) 
        @fastmath n = 1+(n0-1)*exp(-αn*(t-t0)) 
        @fastmath q = 1+(q0-1)*exp(-αq*(t-t0)) 
    
    else # (backward equations, closing of the channels)
        @fastmath m = m0*exp(-βm*(t-t0))
        @fastmath h = 1+(h0-1)*exp(-αh*(t-t0))  
        @fastmath n = n0*exp(-βn*(t-t0))
        @fastmath q = q0*exp(-βq*(t-t0))
    
    end
    
    
    ### CA CHANNELS , for obtaining hysteresis
    
    if vs > Vth && (t1 == -1 ) # Initiate Ca spikes with somatic AP
        SPIKE1 = 1
        t1 = t
        p0 = p
        
    elseif (vs > Vth - VthCa)  && (t1 != -1) && (t - t1) > rp # prolongation of the firing, hysteresis
        SPIKE1 = 1
        t1 = t
        p0 = p
    
    elseif SPIKE1 == 1 && (t - t1) > pd # end of the Ca AP
        SPIKE1 = 0
        t1 = t  
        p0 = p
    
    elseif flag==true && last_spike==true #facilitation effect, start firing again if we are in the ramp down of the pulse and conditions for firing are met
        t1 = -1.0     
    end
    
    
    if SPIKE1 > 0
        @fastmath p = 1 + (p0 - 1) * exp(-αp * (t - t1))
    else
        @fastmath p = p0 * exp(-βp * (t - t1))
    end
    
    
    
    @fastmath new_vs = (1/Cs)*(-gc*(vs-vd)-gls*(vs-EL)-gNa*m^3*h*(vs-ENa)-gKf*n^4*(vs-EK)-gKs*q^2*(vs-EK))
    
    @fastmath new_vd = (1/Cd)*(-gc*(vd-vs)-gld*(vd-EL)-γh*gCa*p*(vd-ECa)-exc_input*Ge*(vd-Ee)- IF*inh_input*Gi*(vd-Ei))
    
    return new_vd, new_vs, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1

end
    
    
"""
    ode4(tspan, vd0, vs0, MN_i, state_params, exc_input, inh_input, MNs)

Fixed-step 4th-order Runge–Kutta integration of a single motoneuron (`MN_i`) over `tspan`.
Unpacks that neuron's parameters from `MNs[MN_i]`, then integrates the two-compartment
model (`ODE_motoneuron`) step by step and records the full time series.

# Arguments
- `tspan`         : time grid (ms).
- `vd0`, `vs0`    : initial dendrite / soma voltages.
- `MN_i`          : index of the motoneuron in `MNs`.
- `state_params`  : initial gating / spike-timing state (14-tuple).
- `exc_input`, `inh_input` : excitatory and inhibitory drive (indexed per ms).
- `MNs`           : list of motoneuron parameter dictionaries.

# Returns
`(Vd, Vs, state_params, Pt, Mt, Ht, Nt, Qt)` — dendrite and soma voltage traces, the final
state tuple, and the gating traces (Ca `p`, Na `m`/`h`, K `n`/`q`).
"""
function ode4(tspan, vd0, vs0, MN_i, state_params, exc_input,inh_input, MNs)

@unpack gc, gld, gls, gCa, Cd, Cs,  αp, βp, VthCa, gNa, αm, βm, Vth, gKf, gKs, αh, βh, αn, βn,αq, βq, γh,IF= stringdict(MNs[MN_i])  

m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = state_params

N = length(tspan)-1 #number of steps
# create variables for everything in case we want to see the evolution over time of each parameter
Vd = zeros(N) 
Vs = zeros(N)
Pt=zeros(N)
Mt=zeros(N)
Ht=zeros(N)
Nt=zeros(N)
Qt=zeros(N)

Vd[1] = vd0
Vs[1] = vs0
Pt[1] = p0
Mt[1] = m0
Ht[1] = h0
Nt[1] = n0
Qt[1] = q0


e_inp_prev=0
last_spike= false
for i in 2:N #computation for time step
    ti = tspan[i-1] #previous time point
    hi = 0.05 #step size por integration, how far we look again in each interval
    vd = Vd[i-1] #voltage dendrite-value for the previous time step
    vs = Vs[i-1] #voltage soma-value for the previous time step
    e_inp = exc_input[_int(ti+1)] 
    i_inp = inh_input[_int(ti+1)]
    
    if e_inp<e_inp_prev
        flag==true
    end

    # solve differential equations

    vd1, vs1, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti, vd, vs, gc, gld, gls, gCa, Cd, Cs, αp,  βp, VthCa, gNa, αm, βm, Vth, gKf, gKs, αh, βh, αn, βn,αq, βq,γh, IF, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp,flag,last_spike)
    vd2, vs2, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti+0.5*hi,vd+0.5*hi*vd1, vs+0.5*hi*vs1, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,gKs, αh, βh, αn, βn, αq, βq, γh, IF, m, h , n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp,flag,last_spike)
    vd3, vs3, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti+0.5*hi,vd+0.5*hi*vd2, vs+0.5*hi*vs2, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,gKs, αh, βh, αn, βn, αq, βq, γh, IF, m,h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp,flag,last_spike)
    vd4, vs4, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(tspan[i],vd+hi*vd3, vs+hi*vs3, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,gKs, αh, βh, αn, βn, αq, βq, γh, IF, m,h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp,flag,last_spike)

    #to compute voltage for the next step
    Vd[i] = vd + (hi/6)*(vd1+2*vd2+2*vd3+vd4) #vd1 and vd4 are given a weight of 1 and vd2 and vd3 a weight of 2 because they are more reliable
    Vs[i] = vs + (hi/6)*(vs1+2*vs2+2*vs3+vs4)
    Pt[i] = p
    Mt[i] = m
    Ht[i] = h
    Nt[i] = n
    Qt[i] = q

    e_inp_prev= e_inp

    # FACILITATION PART 
    if (vs> (Vth-VthCa)) && (Vs[i]> (Vth-VthCa))
        last_spike=false
    elseif (vs> (Vth-VthCa)) && (Vs[i]< (Vth-VthCa))
        last_spike=true
    end


end
state_params = (m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1)
return Vd, Vs, state_params,Pt, Mt, Ht, Nt, Qt
end


"""
    spikes_from_aps(voltage_time_series, i, MNs)

Convert a somatic voltage trace into a binary spike train: a spike (`1`) is marked at each
sample that is (a) above the neuron's threshold `MNs[i][:Vth]` and (b) a local maximum
(greater than both neighbours). Returns an `Int` vector the same length as the input.
"""
function spikes_from_aps(voltage_time_series, i, MNs)
spikes = zeros(Int64, length(voltage_time_series))
for t = 2:length(voltage_time_series)-1

    cond1 = voltage_time_series[t] > MNs[i][:Vth]
    cond2 = voltage_time_series[t-1] < voltage_time_series[t] > voltage_time_series[t+1]
    spikes[t]= _int(cond1 && cond2)
end
return spikes
end

