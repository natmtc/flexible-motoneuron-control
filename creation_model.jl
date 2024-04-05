using Parameters

"""
ODE_motoneuron
Computation of the state variables: ϕ̊ = α(1-ϕ) - βϕ
Computation of the soma and dendrite membrane potentials using the previous calculated stated variables, etc.
"""
function ODE_motoneuron(t::Float64, vd::Float64, vs::Float64,
gc::Float64, gld::Float64, gls::Float64, gCa::Float64, Cd::Float64,
Cs::Float64,  αp::Float64,  βp::Float64, VthCa::Float64, gNa::Float64,
αm::Float64, βm::Float64, Vth::Float64, gKf::Float64, gKs::Float64, αh::Float64, βh::Float64,
αn::Float64, βn::Float64, αq::Float64, βq::Float64, γh::Float64,IF::Float64, m::Float64, h::Float64, 
n::Float64, q::Float64, p::Float64, m0::Float64, h0::Float64, n0::Float64, q0::Float64, p0::Float64,
t0::Float64, t1::Float64, SPIKE0::Int64, SPIKE1::Int64, exc_input,inh_input)


if vs > Vth && (t0 == -1 || (t-t0) > rp)   #initiation of the AP
    SPIKE0 = 1; t0=t; m0=m; h0=h; n0=n;q0=q;p0=p

elseif SPIKE0 == 1 && (t-t0) > pd    #end of the AP
    SPIKE0 = 0; t0=t; m0=m; h0=h; n0=n; q0=q;p0=p

end

if SPIKE0 > 0 #AP (forward equations)

    @fastmath m = 1+(m0-1) * exp(-αm*(t-t0)) 
    @fastmath h = h0*exp(-βh*(t-t0)) 
    @fastmath n = 1+(n0-1)*exp(-αn*(t-t0)) 
    @fastmath q = 1+(q0-1)*exp(-αq*(t-t0)) 

else # (backward equations)
    @fastmath m = m0*exp(-βm*(t-t0))
    @fastmath h = 1+(h0-1)*exp(-αh*(t-t0))  
    @fastmath n = n0*exp(-βn*(t-t0))
    @fastmath q = q0*exp(-βq*(t-t0))

end

if vs > Vth - VthCa && (t1 == -1 || (t-t1) > 0.05)   #calcium-specific threshold (VthaCa)  
    SPIKE1 = 1; t1=t; p0=p;
end
if SPIKE1 == 1 && (t-t1) > pd   
    SPIKE1 = 0; t1=t; p0=p;
end

if SPIKE1 > 0
    @fastmath p = 1+(p0-1) * exp(-αp*(t-t1))
else
    @fastmath p = p0*exp(-βp*(t-t1))

end

@fastmath new_vs = (1/Cs)*(-gc*(vs-vd)-gls*(vs-EL)-gNa*m^3*h*(vs-ENa)-gKf*n^4*(vs-EK)-gKs*q^2*(vs-EK))

@fastmath new_vd = (1/Cd)*(-gc*(vd-vs)-gld*(vd-EL)-γh*gCa*p*(vd-ECa)-exc_input*Ge*(vd-Ee)- IF*inh_input*Gi*(vd-Ei))

return new_vd, new_vs, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1
end




"""
ode4(tspan, vd0, vs0, MN_i, state_params, exc_input,inh_input)
Computes voltage values for the soma and the dendrite for each time step
"""
function ode4(tspan, vd0, vs0, MN_i, state_params, exc_input,inh_input, MNs)
# unpack the variables that are in the MN list for one specififc motoneuron i

@unpack gc, gld, gls, gCa, Cd, Cs,  αp, βp, VthCa, gNa, αm, βm, Vth, gKf, gKs, αh, βh, αn, βn,
αq, βq, γh,IF= stringdict(MNs[MN_i])  

# state parameters

m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = state_params

N = length(tspan)-1 #number of steps
Vd = zeros(N) 
Vs = zeros(N)
Vd[1] = vd0
Vs[1] = vs0

for i in 2:N #computation for time step
    ti = tspan[i-1] #previous time point
    hi = 0.05 #step size por integration, how far we look again in each interval
    vd = Vd[i-1] #voltage dendrite-value for the previous time step
    vs = Vs[i-1] #voltage soma-value for the previous time step
    e_inp = exc_input[_int(ti+1)] 
    i_inp = inh_input[_int(ti+1)]
    
    #Fourth-order Runge-Kutta method that solves differential equations with step-size of 0.05ms.

    vd1, vs1, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti, vd, vs, gc,
    gld, gls, gCa, Cd, Cs, αp,  βp, VthCa, gNa, αm, βm, Vth, gKf, gKs, αh, βh, αn, βn,
    αq, βq,γh, IF, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp)

    vd2, vs2, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti+0.5*hi,
    vd+0.5*hi*vd1, vs+0.5*hi*vs1, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,
    gKs, αh, βh, αn, βn, αq, βq, γh, IF, m, h , n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp)

    vd3, vs3, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(ti+0.5*hi,
    vd+0.5*hi*vd2, vs+0.5*hi*vs2, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,
    gKs, αh, βh, αn, βn, αq, βq, γh, IF, m,h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp)

    vd4, vs4, m, h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1 = ODE_motoneuron(tspan[i],
    vd+hi*vd3, vs+hi*vs3, gc, gld, gls, gCa, Cd, Cs,  αp,  βp, VthCa, gNa, αm, βm, Vth, gKf,
    gKs, αh, βh, αn, βn, αq, βq, γh, IF, m,h, n, q, p, m0, h0, n0, q0, p0, t0, t1, SPIKE0, SPIKE1, e_inp,i_inp)

    #to compute voltage for the next step
    Vd[i] = vd + (hi/6)*(vd1+2*vd2+2*vd3+vd4) #vd1 and vd4 are given a weight of 1 and vd2 and vd3 a weight of 2 because they are more reliable
    Vs[i] = vs + (hi/6)*(vs1+2*vs2+2*vs3+vs4)
end
return Vd, Vs, state_params
end


"""
spikes_from_aps(time_series, i) 
Returns spikes from the soma based on wheter the neuron voltage exceeds a specific threshold at a given time point (voltage t > threshold???)
* voltage_time_series = soma voltage returned by 'ode4' function (see function that runs simulation)
"""
function spikes_from_aps(voltage_time_series, i, MNs)
spikes = zeros(Int64, length(voltage_time_series))
for t = 2:length(voltage_time_series)-1
    # check if voltage is bigger than threshold (Vth)
    # Generate 1 or 0 from Boolean dependending on whether the element meets both conditions
    cond1 = voltage_time_series[t] > MNs[i][:Vth]
    cond2 = voltage_time_series[t-1] < voltage_time_series[t] > voltage_time_series[t+1]
    spikes[t]= _int(cond1 && cond2)
end
return spikes
end


