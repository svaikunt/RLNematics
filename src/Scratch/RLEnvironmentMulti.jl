using ReinforcementLearning
using Random
using IntervalSets

include("SimMain.jl")
using .SimMain

export NematicEnvMulti

########################################################################################
#
#                    RL environment definitions
#
########################################################################################

    # model is https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningEnvironments/src/environments/examples/SpeakerListenerEnv.jl

mutable struct NematicEnvMulti <: AbstractEnv 
    sP::SimParams
    sS::SimState
    fLP::ForceLawParams
    done::Bool
    t::Int 
    nStates::Int 
    nActions::Int
    oneDBool::Bool
    bounds::Vector{Real} # range of increments to activity coefficients
    rewP::Real
    rewTm::Real
    rewTp::Real
    nSteps::Int # max number of steps in RL episode
    updateOncePerStepBool::Bool # whether to update agents and activityField each step (true) or each dt (false), for efficiency
end

function NematicEnvMulti(sP, fLP; # overloaded constructor 
    bounds = ones(6), # used in a tanh clamp from the NN outputs
    rewP = 1.0,
    rewTm = 0, # not used yet
    rewTp = 0, # not used yet
    nSteps = 100,
    oneDBool = false, # whether the include y / theta coefficients in action space
    updateOncePerStepBool = false
    )  
    rng = Random.GLOBAL_RNG

    sS = InitializeSimState(sP) 

    if oneDBool
        nActions = 3
    else 
        nActions = 6
    end

    env = NematicEnvMulti( # call default constructor  
        sP, 
        sS,
        fLP,
        false,
        0, # t
        4, # nStates
        nActions,
        oneDBool,
        bounds, 
        rewP, 
        rewTm, 
        rewTp,
        nSteps,
        updateOncePerStepBool
    )

    reset!(env)

    return env 
end

function RLBase.reset!(env::NematicEnvMulti)

    env.t = 0

    env.sS = InitializeSimState(env.sP)

end

function RLBase.is_terminated(env::NematicEnvMulti) 
    if (! CheckDefectCount(env.sS))
        return true
    elseif (env.t > env.nSteps)
        return true
    else 
        return false 
    end 
end 

RLBase.players(env::NematicEnvMulti) = (:PlusDefect, :MinusDefect)

function GetState(agentHandler)
    rpm = agentHandler.MinusDefects[1].Position .- agentHandler.PlusDefects[1].Position
    return [rpm..., agentHandler.PlusDefects[1].Orientation, agentHandler.MinusDefects[1].Orientation]
end

RLBase.state(env::NematicEnvMulti, ::Observation{Any}, players::Tuple) = Dict(p => state(env, p) for p in players)

function RLBase.state(env::NematicEnvMulti, ::Observation{Any}, player::Symbol)
    if CheckDefectCount(env.sS)
        return GetState(env.sS.agentHandler)
    else 
        return zeros(env.nStates)
    end 
end

RLBase.state_space(env::NematicEnvMulti, ::Observation{Any}, players::Tuple) = 
    Space(Dict(player => state_space(env, player) for player in players)) 

RLBase.state_space(env::NematicEnvMulti, ::Observation{Any}, player::Symbol) = Space(vcat(
    Space([ClosedInterval(-Inf, Inf) for _ in 1:env.nStates])...)) # space of vectors of nStates numbers

    
RLBase.action_space(env::NematicEnvMulti, players::Tuple) = 
    Space(Dict(player => action_space(env, player) for player in players)) 

RLBase.action_space(env::NematicEnvMulti, player::Symbol) = Space(vcat(
    Space([ClosedInterval(-env.bounds[a], env.bounds[a]) for a in 1:env.nActions])...)) # space of vectors of nStates numbers


function CheckDefectCount(sS)
    if (length(sS.agentHandler.PlusDefects) == 1) && (length(sS.agentHandler.MinusDefects) == 1)
        return true
    else 
        return false
    end 
end

function _step!(env::NematicEnvMulti) # wrap SimStep and check number of defects
    SimStep!(env.sS, env.sP, 0, env.updateOncePerStepBool)
    env.t += 1
end
    
function (env::NematicEnvMulti)(actions::Dict, players::Tuple) # call for both players, executes step at the end

    @assert length(actions) == length(players)
    #println(actions)
    for p in players
        env(actions[p], p)
    end
    _step!(env)
end

function (env::NematicEnvMulti)(action::Vector, player::Symbol) # call for individual player, wraps function
    UpdateAgentHandlerFromAction!(env.sS, action, player, env.bounds, env.oneDBool)
end

RLBase.current_player(env::NematicEnvMulti) = (:PlusDefect, :MinusDefect)

function RLBase.reward(env::NematicEnvMulti, player) 

    if (! CheckDefectCount(env.sS))
        predictedState = PredictedState(GetState(env.sS.lastAgentHandler)..., env.sP.ndt, env.fLP)
        currentState = GetState(env.sS.lastAgentHandler) # if annihilation or creation of defects happened, consider "current state" as that before defect number changed
        delrpm = predictedState[1:2] .- currentState[1:2]
        return - env.rewP * (delrpm[1]^2 + delrpm[2]^2) 
    else
        predictedState = PredictedState(GetState(env.sS.lastAgentHandler)..., env.sP.ndt, env.fLP)
        currentState = GetState(env.sS.agentHandler)
        delrpm = predictedState[1:2] .- currentState[1:2]
        return - env.rewP * (delrpm[1]^2 + delrpm[2]^2) # penalize the distance between predicted rpm and current rpm
    end
end 


RLBase.NumAgentStyle(::NematicEnvMulti) = MultiAgent(2)
RLBase.DynamicStyle(::NematicEnvMulti) = SIMULTANEOUS
RLBase.ActionStyle(::NematicEnvMulti) = MINIMAL_ACTION_SET
RLBase.ChanceStyle(::NematicEnvMulti) = DETERMINISTIC