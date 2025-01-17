using ReinforcementLearning
using Random
using IntervalSets

export ToyEnv

########################################################################################
#
#                    RL environment definitions
#
########################################################################################

    # model is https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningEnvironments/src/environments/examples/SpeakerListenerEnv.jl

# struct ToySimParams
#     kEl::Real 
#     pThresh::Real 
#     randBool::Bool 
#     pInitArg::Any 
#     ndt::Int
# end

# mutable struct TweezerParams 
#     rpOff::Real 
#     kp::Real
#     rpEq::Real 
#     rmOff::Real 
#     km::Real 
#     rmEq::Real 
# end

# mutable struct ToySimState
#     p::Real # separation between points
#     p2::Real # separation between points
#     lastp::Real 
#     tweezerParams::TweezerParams
# end

# struct ToyForceLawParams 
#     kst::Real
#     pst::Real
# end 

# function InitializeToySimState(sP, rng)
#     if sP.randBool
#         pInit = sP.pInitArg[1] * (rand(rng) - 0.5) + sP.pInitArg[2]
#     else 
#         pInit = sP.pInitArg 
#     end 
#     return ToySimState(pInit - 5, pInit, pInit, TweezerParams(0.0, 1e-1, 0.0, 0.0, 1e-1, 0.0))
# end

# mutable struct ToyEnv <: AbstractEnv 
#     sP::ToySimParams
#     sS::ToySimState
#     fLP::ToyForceLawParams
#     done::Bool
#     t::Int 
#     nStates::Int 
#     nActions::Int
#     bounds::Vector{Real} # range of increments to activity coefficients
#     rewP::Real
#     nSteps::Int # max number of steps in RL episode
#     extraActionsBool::Bool
#     reward::Real
#     rng::Any 
#     episodeCount::Int
# end

# function ToyEnv(sP, fLP, rng; # overloaded constructor 
#     bounds = ones(6), # used in a tanh clamp from the NN outputs
#     rewP = 1.0,
#     nSteps = 100,
#     extraActionsBool = false, 
#     )  

#     sS = InitializeToySimState(sP, rng) 

#     if !extraActionsBool
#         nActions = 1
#     else 
#         nActions = 2
#     end 
    
#     env = ToyEnv( # call default constructor  
#         sP, 
#         sS,
#         fLP,
#         false,
#         0, # t
#         1, # nStates
#         nActions,
#         bounds, 
#         rewP, 
#         nSteps,
#         extraActionsBool, 
#         0.0, 
#         rng,
#         0
#     )

#     reset!(env)

#     return env 
# end

# function RLBase.reset!(env::ToyEnv)

#     env.t = 0
#     env.reward = 0.0
#     env.done = false

#     env.sS = InitializeToySimState(env.sP, env.rng)

# end

# function RLBase.is_terminated(env::ToyEnv) 
#     if (env.t > env.nSteps)
#         return true
#     else 
#         return false 
#     end 
# end 

# RLBase.players(env::ToyEnv) = (:PlusDefect, :MinusDefect)

# RLBase.state(env::ToyEnv, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)

# function RLBase.state(env::ToyEnv, ::Observation{Any}, player::Symbol)
#     if player == :MinusDefect 
#         return [env.sS.p2]
#     else 
#         return [env.sS.p]
#     end
#     #[env.sS.p, env.sS.p2]
# end

# RLBase.state_space(env::ToyEnv, ::Observation{Any}, players::Tuple) = 
#     Space(Dict(player => state_space(env, player) for player in players)) 

# RLBase.state_space(env::ToyEnv, ::Observation{Any}, player::Symbol) = Space(vcat(
#     Space([ClosedInterval(-Inf, Inf) for _ in 1:env.nStates])...)) # space of vectors of nStates numbers

    
# RLBase.action_space(env::ToyEnv, players::Tuple) = 
#     Space(Dict(player => action_space(env, player) for player in players)) 

# RLBase.action_space(env::ToyEnv, player::Symbol) = Space(vcat(
#     Space([ClosedInterval(-1, 1) for a in 1:env.nActions])...)) # space of vectors of nStates numbers


# function CheckCollision(sS, sP)
#     if sS.p < sP.pThresh
#         return true
#     else 
#         return false
#     end 
# end

# function _step!(env::ToyEnv) 

#     env.sS.lastp = deepcopy(env.sS.p2)

#     for _ in 1:env.sP.ndt
#         #env.sS.p += -2 * env.sP.kEl * env.sS.p + env.sS.tweezerParams.km * (env.sS.tweezerParams.rmOff - env.sS.tweezerParams.rmEq) - env.sS.tweezerParams.kp * (env.sS.tweezerParams.rpOff - env.sS.tweezerParams.rpEq)
#         env.sS.p += env.sP.kEl * (env.sS.p2 - env.sS.p) + env.sS.tweezerParams.kp * (env.sS.tweezerParams.rpOff - env.sS.tweezerParams.rpEq)
#         env.sS.p2 += -env.sP.kEl * (env.sS.p2 - env.sS.p) + env.sS.tweezerParams.km * (env.sS.tweezerParams.rmOff - env.sS.tweezerParams.rmEq)
#     end 
    
#     env.t += 1
# end

# # function PredictedToyState(p, delt, fLP)

# #     return p + delt * (- fLP.kst * (p - fLP.pst)) 
# # end 

    
# function (env::ToyEnv)(actions::Dict, players::Tuple) # call for both players, executes step at the end

#     @assert length(actions) == length(players)
#     for pl in players
#         env(actions[pl], pl)
#     end
#     _step!(env)
# end

# function (env::ToyEnv)(action::Vector, player::Symbol) # call for individual player, wraps function

#     # if env.extraActionsBool
#     #     env.sS.tweezerParams.r1Eq = bounds[5] * tanh(action[5])
#     #     env.sS.tweezerParams.r2Eq = bounds[6] * tanh(action[6])
#     # end 

#     if player == :PlusDefect 
#         env.sS.tweezerParams.rpOff = bounds[1] * action[1]
#         #env.sS.tweezerParams.kp = 1e-1 #bounds[2] * (tanh(action[2]) + 1)
#     elseif player == :MinusDefect 
#         env.sS.tweezerParams.rmOff = bounds[3] * action[1]
#         #env.sS.tweezerParams.km = 1e-1 #bounds[4] * (tanh(action[2]) + 1)
#     else 
#         println("Unrecognized player")
#     end

# end

# function (env::ToyEnv)(action::Real, player::Symbol) # call for individual player, wraps function

#     if player == :PlusDefect 
#         env.sS.tweezerParams.rpOff = bounds[1] * action
#         #env.sS.tweezerParams.kp = 1e-1 #bounds[2] * (tanh(action[2]) + 1)
#     elseif player == :MinusDefect 
#         env.sS.tweezerParams.rmOff = bounds[3] * action
#         #env.sS.tweezerParams.km = 1e-1 #bounds[4] * (tanh(action[2]) + 1)
#     else 
#         println("Unrecognized player")
#     end

# end

# RLBase.current_player(env::ToyEnv) = (:PlusDefect, :MinusDefect)

# function RLBase.reward(env::ToyEnv, player) 

#     # if abs(env.sS.p - env.fLP.pst) > 4
#     #     return - env.rewP * (5 + abs.(env.sS.p - env.fLP.pst)^(1/2) ) 
#     # elseif abs(env.sS.p - env.fLP.pst) < 1
#     #     return - env.rewP * (-5 + abs.(env.sS.p - env.fLP.pst)^(1/2) ) 
#     # else
#     #     return - env.rewP * abs.(env.sS.p - env.fLP.pst)^(1/2)
#     # end 


#     #return - env.rewP * ( abs.(env.sS.p - env.fLP.pst)^(1/2) )  + 5 * env.rewP * (env.sS.p2 - env.sS.lastp)

#     rewEx = 0.5 * (tanh((env.episodeCount - 1250) / 250) + 1)
#     #return  env.rewP * ( (env.sS.p2 - env.sS.lastp) - rewEx * abs(env.sS.p2 - env.sS.p - env.fLP.pst) / 10 )
#     return - env.rewP * (abs(env.sS.p2 - env.fLP.pst) + abs(env.sS.p))
#     #return env.rewP * (env.sS.p - env.sS.lastp)

# end 


# RLBase.NumAgentStyle(::ToyEnv) = MultiAgent(2)
# RLBase.DynamicStyle(::ToyEnv) = SIMULTANEOUS
# RLBase.ActionStyle(::ToyEnv) = MINIMAL_ACTION_SET
# RLBase.ChanceStyle(::ToyEnv) = DETERMINISTIC
# RLBase.UtilityStyle(::ToyEnv) = IDENTICAL_UTILITY
# RLBase.InformationStyle(::ToyEnv) = IMPERFECT_INFORMATION






struct ToySimParams
    kEl::Real 
    pThresh::Real 
    randBool::Bool 
    pInitArg::Any 
    ndt::Int
end

mutable struct TweezerParams 
    rpOff::Real 
    kp::Real
    rpEq::Real 
    rmOff::Real 
    km::Real 
    rmEq::Real 
end

mutable struct ToySimState
    p::Real # separation between points
    p2::Real 
    lastp::Real 
    tweezerParams::TweezerParams
end

struct ToyForceLawParams 
    kst::Real
    pst::Real
end 

function InitializeToySimState(sP, rng)
    if sP.randBool
        pInit = sP.pInitArg[1] * (rand(rng) - 0.5) + sP.pInitArg[2]
    else 
        pInit = sP.pInitArg 
    end 
    return ToySimState(0, pInit, pInit, TweezerParams(0.0, 1e-1, 0.0, 0.0, 1e-1, 0.0))
end

mutable struct ToyEnv <: AbstractEnv 
    sP::ToySimParams
    sS::ToySimState
    fLP::ToyForceLawParams
    done::Bool
    t::Int 
    nStates::Int 
    nActions::Int
    bounds::Vector{Real} # range of increments to activity coefficients
    rewP::Real
    nSteps::Int # max number of steps in RL episode
    extraActionsBool::Bool
    reward::Real 
    rng::Any
    episodeCount::Int
end

function ToyEnv(sP, fLP, rng; # overloaded constructor 
    bounds = ones(6), # used in a tanh clamp from the NN outputs
    rewP = 1.0,
    nSteps = 100,
    extraActionsBool = false, 
    )  
 
    sS = InitializeToySimState(sP, rng) 

    if !extraActionsBool
        nActions = 2
    else 
        nActions = 3
    end 
    
    env = ToyEnv( # call default constructor  
        sP, 
        sS,
        fLP,
        false,
        0, # t
        1, # nStates
        nActions,
        bounds, 
        rewP, 
        nSteps,
        extraActionsBool, 
        0.0,
        rng,
        0
    )

    reset!(env)

    return env 
end

function RLBase.reset!(env::ToyEnv)

    env.t = 0
    env.reward = 0.0

    env.sS = InitializeToySimState(env.sP, env.rng)

end

function RLBase.is_terminated(env::ToyEnv) 
    # if CheckCollision(env.sS, env.sP)
    #     return true
    if (env.t > env.nSteps)
        return true
    else 
        return false 
    end 
end 


function RLBase.state(env::ToyEnv)
    #return [(env.sS.p - env.fLP.pst) / (env.sP.pInitArg[1])]
    return [(env.sS.p2 - env.sS.p - env.fLP.pst) / (env.sP.pInitArg[1])]
end

RLBase.state_space(env::ToyEnv, ::Observation{Any}) = Space(vcat(
    Space([ClosedInterval(-Inf, Inf) for _ in 1:env.nStates])...)) # space of vectors of nStates numbers

RLBase.action_space(env::ToyEnv) = Space(vcat(
    Space([ClosedInterval(-env.bounds[a], env.bounds[a]) for a in 1:env.nActions])...)) # space of vectors of nStates numbers


function CheckCollision(sS, sP)
    if sS.p < sP.pThresh
        return true
    else 
        return false
    end 
end

function _step!(env::ToyEnv) 

    env.sS.lastp = deepcopy(env.sS.p2 - env.sS.p)

    for _ in 1:env.sP.ndt
        #env.sS.p += -2 * env.sP.kEl * env.sS.p + env.sS.tweezerParams.km * (env.sS.tweezerParams.rmOff - env.sS.tweezerParams.rmEq) - env.sS.tweezerParams.kp * (env.sS.tweezerParams.rpOff - env.sS.tweezerParams.rpEq)
        env.sS.p += env.sP.kEl * env.sS.p - env.sS.tweezerParams.kp * (env.sS.tweezerParams.rpOff - env.sS.tweezerParams.rpEq)
        env.sS.p2 += - env.sP.kEl * env.sS.p + env.sS.tweezerParams.km * (env.sS.tweezerParams.rmOff - env.sS.tweezerParams.rmEq)
    end 

    env.t += 1
end


function (env::ToyEnv)(action::Real) # call for both players, executes step at the end
    env.sS.tweezerParams.rpOff = bounds[1] * action
    _step!(env)
end

function (env::ToyEnv)(action::Vector) # call for both players, executes step at the end
    env.sS.tweezerParams.rmOff = bounds[1] * action[1]
    env.sS.tweezerParams.rpOff = bounds[3] * action[2]
    _step!(env)
end


function PredictedToyState(p, delt, fLP)

    return p + delt * (- fLP.kst * (p - fLP.pst)) 
end 

function RLBase.reward(env::ToyEnv) 

    # if CheckCollision(env.sS, env.sP)
    #     env.reward = - env.rewP * (env.sS.lastp - env.fLP.pst)^2 
    # else 
        #env.reward = - env.rewP * abs(env.sS.p - env.fLP.pst)
        #rewEx = 0.5 * (tanh((env.episodeCount - 1250) / 250) + 1)
        #env.reward = env.rewP * (- abs(env.sS.p2 - env.sS.p - env.fLP.pst) + 1 * (env.sS.p2 - env.sS.lastp))
        #env.reward =  env.rewP * (env.sS.p - env.sS.lastp) 
    #end

    predictedp = PredictedToyState(env.sS.lastp, env.sP.ndt, env.fLP)
    currentp = env.sS.p2 - env.sS.p
    env.reward = - env.rewP * abs(currentp - predictedp)
end 








# struct PendulumEnvParams
#     max_speed::Real
#     max_torque::Real
#     g::Real
#     m::Real
#     l::Real
#     dt::Real
#     max_steps::Int
# end


# mutable struct ToySimState
#     t::Real # separation between points
#     tdot::Real 
# end


# mutable struct ToyEnv <: AbstractEnv 
#     sP::PendulumEnvParams
#     sS::ToySimState
#     torque::Real
#     done::Bool
#     t::Int 
#     rng::AbstractRNG
#     nStates::Int 
#     nActions::Int
#     reward::Real 
# end

# function ToyEnv(;
#     T = Float64,
#     max_speed = T(8),
#     max_torque = T(2),
#     g = T(10),
#     m = T(1),
#     l = T(1),
#     dt = T(0.05),
#     max_steps = 200,
#     rng = Random.GLOBAL_RNG
# )
#     env = ToyEnv(
#         PendulumEnvParams(max_speed, max_torque, g, m, l, dt, max_steps),
#         ToySimState(0.0, 0.0),
#         zero(T),
#         false,
#         0,
#         rng,
#         3, 
#         1,
#         zero(T)
#     )
#     reset!(env)
#     env
# end

# pendulum_observation(s) = [cos(s.t), sin(s.t), s.tdot]
# angle_normalize(x) = Base.mod((x + Base.π), (2 * Base.π)) - Base.π

# RLBase.action_space(env::ToyEnv) = -2.0..2.0 
# RLBase.state_space(env::ToyEnv) = Space(ClosedInterval{T}.(-T.([1, 1, env.sP.max_speed]), T.([1, 1, env.sP.max_speed])))
# RLBase.reward(env::ToyEnv) = env.reward
# RLBase.is_terminated(env::ToyEnv) = env.done
# RLBase.state(env::ToyEnv) = pendulum_observation(env.sS)

# function RLBase.reset!(env::ToyEnv)

#     T = Float64

#     env.sS.t = 2 * π * (rand(env.rng, T) .- 1)
#     env.sS.tdot = 2 * (rand(env.rng, T) .- 1)
#     env.torque = zero(T)
#     env.t = 0
#     env.done = false
#     env.reward = zero(T)
#     nothing

# end

# function (env::ToyEnv)(a::Real)
#     low = -2
#     high = 2
    
#     env.torque = low + (a + 1) * 0.5 * (high - low)
#     _step!(env, env.torque)
# end


# function _step!(env::ToyEnv, a)   # call for both players, executes step at the end
#     env.t += 1
#     th = env.sS.t
#     thdot = env.sS.tdot
#     a = clamp(a, -env.sP.max_torque, env.sP.max_torque)
#     costs = angle_normalize(th)^2 + 0.1 * thdot^2 + 0.001 * a^2
#     newthdot =
#         thdot +
#         (
#             -3 * env.sP.g / (2 * env.sP.l) * sin(th + pi) +
#             3 * a / (env.sP.m * env.sP.l^2)
#         ) * env.sP.dt
#     th += newthdot * env.sP.dt
#     newthdot = clamp(newthdot, -env.sP.max_speed, env.sP.max_speed)
#     env.sS.t = th
#     env.sS.tdot = newthdot
#     env.done = env.t >= env.sP.max_steps
#     env.reward = -costs
#     nothing
# end
