using ReinforcementLearning
using Random
using IntervalSets

include("SimMain.jl")
using .SimMain

include("MathFunctions.jl")
using .MathFunctions

export NematicEnv

########################################################################################
#
#                    RL environment definitions
#
########################################################################################

    # model is https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/2e1de3e5b6b8224f50b3d11bba7e1d2d72c6ef7c/src/ReinforcementLearningEnvironments/src/environments/examples/SpeakerListenerEnv.jl

mutable struct NematicEnv <: AbstractEnv 
    sP::SimParams
    sS::SimState
    fLP::ForceLawParams
    done::Bool
    t::Int 
    taskMarker::String
    nStates::Int 
    nActions::Int
    bounds::Vector{Real} # range of increments to activity coefficients
    rewP::Real
    rewT::Real
    nSteps::Int # max number of steps in RL episode
    updateOncePerStepBool::Bool # whether to update agents and activityField each step (true) or each dt (false), for efficiency
    reward::Real
    rng::Any
    episodeCount::Int
end

function NematicEnv(sP, fLP, rng; # overloaded constructor 
    bounds = ones(4), # used in a tanh clamp from the NN outputs
    taskMarker = "sep",
    rewP = 1.0,
    rewT = 0, # not used yet
    nSteps = 100,
    updateOncePerStepBool = false
    )      

    nActions = length(bounds)
  
    sS = InitializeSimState(sP, rng, fLP, taskMarker, nSteps) 

    if (taskMarker == "sepMin") || (taskMarker == "sepPlus")
        nStates = 1
    elseif taskMarker == "orPlus"
        nStates = 1
    elseif taskMarker == "orPlusPert"
        nStates = 1 #3
    elseif (taskMarker == "orbit") 
        nStates = 2 
    elseif (taskMarker == "follower")
        nStates = 2
    end


    env = NematicEnv( # call default constructor  
        sP, 
        sS,
        fLP,
        false,
        0, # t
        taskMarker,
        nStates, # nStates
        nActions,
        bounds,
        rewP, 
        rewT, 
        nSteps,
        updateOncePerStepBool, 
        0.0,
        rng,
        0
    )

    return env 
end

function RLBase.reset!(env::NematicEnv)

    env.t = 0
    env.reward = 0.0
    env.sS = InitializeSimState(env.sP, env.rng, env.fLP, env.taskMarker, env.nSteps)

end

function RLBase.is_terminated(env::NematicEnv) 
    if (! CheckDefectCount(env.sS))
        return true
    elseif (env.t > env.nSteps)
        return true
    else 
        return false 
    end 
end 

function pmVec(agentHandler) # points from plus to minus
    if (length(agentHandler.PlusDefects) == 1) && (length(agentHandler.MinusDefects) == 1)
        return agentHandler.MinusDefects[1].Position .- agentHandler.PlusDefects[1].Position
    else
        return [0.0, 0.0]
    end
end 


function UpdateAgentHandlerFromAction!(env, action, bounds)

    sS = env.sS

    ### common to all - set r0's to the positions
    sS.agentHandler.MinusDefects[1].activityCoefficients.r0x = sS.agentHandler.MinusDefects[1].Position[1] 
    sS.agentHandler.MinusDefects[1].activityCoefficients.r0y = sS.agentHandler.MinusDefects[1].Position[2]
    sS.agentHandler.PlusDefects[1].activityCoefficients.r0x = sS.agentHandler.PlusDefects[1].Position[1] 
    sS.agentHandler.PlusDefects[1].activityCoefficients.r0y = sS.agentHandler.PlusDefects[1].Position[2] 

    ### for one d minus tweezers - 2 actions
    if env.taskMarker == "sepMin"
        sS.agentHandler.MinusDefects[1].activityCoefficients.r0x += bounds[1] * action[1]
	    sS.agentHandler.MinusDefects[1].activityCoefficients.c0 = 0.5 * bounds[2] * (action[2] + 1)
        #sS.agentHandler.MinusDefects[1].activityCoefficients.c0 = 0.5 * bounds[2] * (action[2] + 1) * 6.5
        #sS.agentHandler.MinusDefects[1].activityCoefficients.cxx = 0.5 * bounds[2] * (action[2] + 1) * -0.05
    
    ### for one d plus tweezer - 1 action
    elseif env.taskMarker == "sepPlus"
        sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = 1 * bounds[1] * action[1]
    
    ### for plus defect rotation - 2 actions
    elseif (env.taskMarker == "orPlus") || (env.taskMarker == "orPlusPert") 

#        sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = 0.5 * bounds[1] * (action[1] + 1.0)

        r = 5
        phiOff = bounds[1] * action[1] + sS.agentHandler.PlusDefects[1].Orientation 
        rOff = [r * cos(phiOff), r * sin(phiOff)]
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0x += rOff[1]
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0y += rOff[2]

        #c0 = 0.5 * bounds[2] * (action[2] + 1) - 5
	#c0 = 0.5 * bounds[2] * (action[2] + 1) 
        #sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = c0
        
    elseif env.taskMarker == "orbit"
        rPlus = 7.5
        cFac = 0.5 * bounds[1] * (action[1] + 1)
        sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = cFac 

        theta = bounds[2] * action[2]
        sS.agentHandler.PlusDefects[1].activityCoefficients.theta = theta

    elseif env.taskMarker == "follower"

        rpm = pmVec(sS.agentHandler)
        rpm ./= sqrt(rpm[1]^2 + rpm[2]^2)
        rMin = bounds[1] * action[1] 
        rOffMin = rMin .* rpm
        sS.agentHandler.MinusDefects[1].activityCoefficients.r0x += rOffMin[1]
        sS.agentHandler.MinusDefects[1].activityCoefficients.r0y += rOffMin[2]

        r = 5.0
        phiOff = bounds[2] * action[2] + sS.agentHandler.PlusDefects[1].Orientation + pi
        rOff = [r * cos(phiOff), r * sin(phiOff)]
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0x += rOff[1]
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0y += rOff[2]

    end 

end

function GetPsiOmegaNearPlusDefect(env)
    (OmegaSoA, PsiSoA) = OmegaPsi2DSoA(env.sP.grid, env.sS.velocitySoA, env.sP.bcBE_X, env.sP.bcBE_Y)
    indPlus = Int.(floor.(env.sS.agentHandler.PlusDefects[1].Position))
    
    avOmega = 0.0
    nx = env.sP.grid.Nx 
    ny = env.sP.grid.Ny
    for ip = 0:1, jp = 0:1
        avOmega += OmegaSoA.XYValues[mod1(indPlus[1] + ip, nx), mod1(indPlus[2] + jp, ny)] / 4
    end

    avPsi= 0.0
    for ip = 0:1, jp = 0:1
        avPsi += PsiSoA.XYValues[mod1(indPlus[1] + ip, nx), mod1(indPlus[2] + jp, ny)] / 4
    end
    
    return (avOmega, avPsi)
end

function GetState(env; last = false) 

    if !last 
        aH = env.sS.agentHandler
    else 
        aH = env.sS.lastAgentHandler
    end

    ### for separation 
    if (env.taskMarker == "sepMin") || (env.taskMarker == "sepPlus") 
        rpm = pmVec(aH)
        sR = sqrt(rpm[1]^2 + rpm[2]^2)
        if (length(aH.PlusDefects) == 1) && (length(aH.MinusDefects) == 1)
            if aH.MinusDefects[1].Position[1] > aH.PlusDefects[1].Position[1]
                sR = env.sP.grid.Nx - sR 
            end
        end
        sS = [2 * (sR - env.fLP.l0) / (env.sP.initParams[1])]
#	 sS = [4 * (sR - env.fLP.l0) / (env.sP.initParams[1])]
        sE = 0
    
    ### for plus end orientation
    elseif (env.taskMarker == "orPlus") 

        plusOr = aH.PlusDefects[1].Orientation
        sR = plusOr
        sS = [sin(plusOr)]
        sE = 0

    elseif (env.taskMarker == "orPlusPert") 
        if (CheckDefectCount(env.sS))
            (avOmega, avPsi) = GetPsiOmegaNearPlusDefect(env) # disregarding last bool

            plusOr = aH.PlusDefects[1].Orientation
            sR = aH.PlusDefects[1].Orientation
            sS = [4 * sin(plusOr)]#, avOmega * 1e1, avPsi * 1e1]
            sE = 0
        else 
            sR = [0.0, 0.0]
            sS = [0.0]
            sE = [0.0]
        end

    elseif (env.taskMarker == "orbit") || (env.taskMarker == "follower")
        if (CheckDefectCount(env.sS))
            rpmO = pmVec(aH)
            rNorm = sqrt(rpmO[1]^2 + rpmO[2]^2)
            rpm = rpmO ./ rNorm
            plusOr = aH.PlusDefects[1].Orientation
            ePlus = [cos(plusOr), sin(plusOr)] 
            dot = ePlus[1] * rpm[1] + ePlus[2] * rpm[2]
            phi = acos(clamp(dot, -1, 1))

            sR = [rNorm, phi]
            sS = [2*(abs(rpmO[1]) - env.fLP.l0) / (env.sP.initParams[1]), sin(plusOr)]
            sE = rpm
        else 
            sR = [0.0, 0.0]
            sS = [0.0, 0.0]
            sE = [0.0, 0.0]
        end
    end

    return (sR, sS, sE) # return raw and scaled version
end

function RLBase.state(env::NematicEnv)
    if CheckDefectCount(env.sS)
        return GetState(env)[2]
    else 
        return zeros(env.nStates)
    end 
end

RLBase.state_space(env::NematicEnv, ::Observation{Any}) = Space(vcat(
    Space([ClosedInterval(-Inf, Inf) for _ in 1:env.nStates])...)) # space of vectors of nStates numbers

RLBase.action_space(env::NematicEnv) = Space(vcat(
    Space([ClosedInterval(-env.bounds[a], env.bounds[a]) for a in 1:env.nActions])...)) # space of vectors of nStates numbers

function CheckDefectCount(sS)
    if (length(sS.agentHandler.PlusDefects) == 1) && (length(sS.agentHandler.MinusDefects) == 1)
        return true
    else 
        return false
    end 
end
    
function (env::NematicEnv)(action::Vector) # call action vector, wraps function
    UpdateAgentHandlerFromAction!(env, action, env.bounds)
    _step!(env)
end

function (env::NematicEnv)(action::Real) # call for single action, wraps function
    UpdateAgentHandlerFromAction!(env, [action], env.bounds)
    _step!(env)
end

function _step!(env::NematicEnv) # wrap SimStep and check number of defects
    SimStep!(env.sS, env.sP, 0, env.updateOncePerStepBool)
    env.t += 1
end

function PredictedState(p, delt, fLP)
    return p + delt * (- fLP.ks * (p - fLP.l0)) 
end 


function RLBase.reward(env::NematicEnv) 

    if (env.taskMarker == "sepMin") || (env.taskMarker == "sepPlus") || (env.taskMarker == "orPlus") 
        if (! CheckDefectCount(env.sS))
            current = GetState(env; last = true)[1]
        else
            current = GetState(env)[1]
        end
        lastp = GetState(env; last = true)[1]
        predicted = PredictedState(lastp, env.sP.ndt, env.fLP)
        #env.reward = - env.rewP * (abs(current - predicted))
	env.reward = -env.rewP * abs(current - env.fLP.l0)

    elseif (env.taskMarker == "orPlusPert")
        if (!CheckDefectCount(env.sS))
            current = sin(env.sS.lastAgentHandler.PlusDefects[1].Orientation)
        else
            current = sin(env.sS.agentHandler.PlusDefects[1].Orientation)
        end
        last =  sin(env.sS.lastAgentHandler.PlusDefects[1].Orientation)

        predicted = PredictedState(last, env.sP.ndt, env.fLP)
        env.reward = - env.rewP * abs(current - predicted)
        
        #env.reward = - env.rewP * abs(sin(orPlus))

    elseif env.taskMarker == "orbit"
        if (! CheckDefectCount(env.sS))
            current = GetState(env; last = true)[1][1]
            currentrpm = GetState(env; last = true)[3]
            rewFail = - 5 * env.rewT
        else
            current = GetState(env)[1][1]
            currentrpm = GetState(env)[3]
            rewFail = 0
        end
        lastp = GetState(env; last = true)[1][1]
        predicted = PredictedState(lastp, env.sP.ndt, env.fLP)
        transRew = - env.rewP * abs(current - predicted)
        
        #lastphi = GetState(env; last = true)[1][2]
        if (! CheckDefectCount(env.sS))
            phi = GetState(env)[1][2]
        else
            phi = 0
        end
        rotRew = - env.rewT * abs(phi - pi/2)

        
        env.reward = transRew + rotRew  + rewFail

    elseif env.taskMarker == "follower"
        if (!CheckDefectCount(env.sS))
            rpm = pmVec(env.sS.lastAgentHandler)
            orPlus = env.sS.lastAgentHandler.PlusDefects[1].Orientation 
        else
            rpm = pmVec(env.sS.agentHandler)
            orPlus = env.sS.agentHandler.PlusDefects[1].Orientation 
        end

        horSep = abs(rpm[1])
        verSep = rpm[2]


        distRew = - env.rewP * abs(horSep - env.fLP.l0) - env.rewT * abs(sin(orPlus))

        
        lastPlusPos = env.sS.lastAgentHandler.PlusDefects[1].Position
        if CheckDefectCount(env.sS)
            plusPos = env.sS.agentHandler.PlusDefects[1].Position
        else 
            plusPos = lastPlusPos 
        end

        velRew = 0 * env.rewT * (plusPos[1] - lastPlusPos[1])

        env.reward = distRew + velRew
    
    end 
end 

