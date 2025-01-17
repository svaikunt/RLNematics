using ReinforcementLearning
using StableRNGs
using Flux
using JLD2
using Flux.Losses
using Flux: params

# model is https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MADDPG_SpeakerListener/#JuliaRL\\_MADDPG\\_SpeakerListener

# see DDPG code at https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl

#  https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/a6623a85110ce5124d575329784dcd3486b2e62a/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/ddpg.jl

# see MADDPG code at https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/main/src/ReinforcementLearningZoo/src/algorithms/policy_gradient/maddpg.jl


########################################################
#
#                    Define Hooks
#
#########################################################


## RLNematic Hook

Base.@kwdef mutable struct EpisodeInformation
    initParams::Vector{Real} = []# store the initial defect configuration
    rewards::Vector{Real} = []
    agentHandlerList::Vector{Any} = []
    
end

Base.@kwdef mutable struct RLNematicHook <: AbstractHook
    episodeList::Vector{EpisodeInformation} = []
    stateTrajList::Vector{Any} = []
    actorLossList::Vector{Real} = []
    criticLossList::Vector{Real} = []
    annealBool::Bool = false
    annealTime::Real = 100

    storingBool::Bool = true # track whether to store episodeInformation during current episode
    stateTrajBool::Bool = true # track whether to store stateTrajectory during current episode
    stepStride::Int = 1 # store every stepStride steps per episode
    stepCount::Int = 0
    episodeStride::Int = 5 # store every episodeStride steps per experiment
    stateTrajStride::Int = 50 # store every episodeStride steps per experiment
    episodeCount::Int = 0
end


function (hook::RLNematicHook)(::PostActStage, agent, env)
    if hook.storingBool && (hook.stepCount % hook.stepStride == 0)
        push!(hook.episodeList[end].rewards, reward(env))
        push!(hook.episodeList[end].agentHandlerList, deepcopy(env.sS.agentHandler))
    end 
    if hook.stateTrajBool && (hook.stepCount % hook.stepStride == 0)
        push!(hook.stateTrajList[end], deepcopy(env.sS))
    end 

    if hook.annealBool 
        agent.policy.act_noise *= (1 - 1 / hook.annealTime)
        agent.policy.act_noise *= (1 - 1 / hook.annealTime)
    end 
    
    hook.stepCount += 1
end

function (hook::RLNematicHook)(::PreEpisodeStage, agent, env)

    GC.gc() # run garbage collection

    if hook.episodeCount % hook.episodeStride == 0
        push!(hook.episodeList, EpisodeInformation())
        hook.episodeList[end].initParams = deepcopy(env.sS.initParams)
        hook.storingBool = true
    else 
        hook.storingBool = false 
    end
    if hook.episodeCount % hook.stateTrajStride == 0
        push!(hook.stateTrajList, [])
        hook.stateTrajBool = true
    else 
        hook.stateTrajBool = false 
    end
    hook.stepCount = 0
end

function (hook::RLNematicHook)(::PostEpisodeStage, agent, env)

    push!(hook.actorLossList, agent.policy.actor_loss)
    push!(hook.criticLossList, agent.policy.critic_loss)

    hook.episodeCount += 1
end

## Toy Hook

Base.@kwdef mutable struct ToyEpisodeInformation
    rewards::Vector{Real} = []
    pList::Vector{Real} = []
    sSList::Vector{Any} = []

end

Base.@kwdef mutable struct RLToyHook <: AbstractHook
    episodeList::Vector{ToyEpisodeInformation} = []
    plusActorLossList::Vector{Real} = []
    plusCriticLossList::Vector{Real} = []
    minusActorLossList::Vector{Real} = []
    minusCriticLossList::Vector{Real} = []
    annealBool::Bool = false
    annealTime::Real = 100

    storingBool::Bool = true # track whether to store information during current episode
    stepStride::Int = 1 # store every stepStride steps per episode
    stepCount::Int = 0
    episodeStride::Int = 5 # store every episodeStride steps per experiment
    episodeCount::Int = 0
end


function (hook::RLToyHook)(::PostActStage, agent, env)
    if hook.storingBool && (hook.stepCount % hook.stepStride == 0)
        push!(hook.episodeList[end].rewards, reward(env))
        #push!(hook.episodeList[end].pList, deepcopy(env.sS.p))
        push!(hook.episodeList[end].sSList, deepcopy(env.sS))
    end 

    # if hook.annealBool 
    #     agent.agents[:PlusDefect].policy.policy.act_noise *= (1 - 1 / hook.annealTime)
    #     agent.agents[:MinusDefect].policy.policy.act_noise *= (1 - 1 / hook.annealTime)
    # end 

    if hook.annealBool 
        agent.policy.act_noise *= (1 - 1 / hook.annealTime)
        agent.policy.act_noise *= (1 - 1 / hook.annealTime)
    end 

    hook.stepCount += 1
end

function (hook::RLToyHook)(::PreEpisodeStage, agent, env)

    #GC.gc() # run garbage collection

    if hook.episodeCount % hook.episodeStride == 0
        push!(hook.episodeList, ToyEpisodeInformation())
        hook.storingBool = true
    else 
        hook.storingBool = false 
    end
    hook.stepCount = 0
end

function (hook::RLToyHook)(::PostEpisodeStage, agent, env)

    # push!(hook.plusActorLossList, agent.agents[:PlusDefect].policy.policy.actor_loss)
    # push!(hook.plusCriticLossList, agent.agents[:PlusDefect].policy.policy.critic_loss)
    # push!(hook.minusActorLossList, agent.agents[:MinusDefect].policy.policy.actor_loss)
    # push!(hook.minusCriticLossList, agent.agents[:MinusDefect].policy.policy.critic_loss)

    push!(hook.plusActorLossList, agent.policy.actor_loss)
    push!(hook.plusCriticLossList, agent.policy.critic_loss)

    push!(hook.minusActorLossList, agent.policy.actor_loss)
    push!(hook.minusCriticLossList, agent.policy.critic_loss)

    hook.episodeCount += 1
    env.episodeCount += 1
end


#############################################################
#
#                    Experiment function 
#
#############################################################

# function CreateMADDPGExperiment(env; # works for toy or the full RL env
#     seed = 0,
#     eps_or_hrs = "eps",
#     envMarker = "nem", 
#     nEpisodes = 10,
#     batchSize = 128,
#     updateFreq = 100,
#     stepStride = 1,
#     episodeStride = 5,
#     netLayers = 3,
#     netWidth = 64,
#     gamma = 0.95f0,
#     rho = 0.99f0,
#     act_limit = 1.0,
#     act_noise = 1e-3,
#     annealBool = false,
#     annealTime = 100
#     )

#     rng = env.rng

#     #rng = StableRNG(seed)

#     init = glorot_uniform(rng) # random NN weight initialization
#     critic_dim = sum(length(state(env, p)) + length(action_space(env, p)) for p in (:PlusDefect, :MinusDefect))
#     #critic_dim = sum(length(state(env, p)) + length(action_space(env, p)) for p in (:Speaker, :Listener))

#     create_actor(player) = Chain( 
#         Dense(length(state(env, player)), netWidth, relu; init = init),
#         [Dense(netWidth, netWidth, relu; init = init) for _ in 1:netLayers]...,
#         Dense(netWidth, length(action_space(env, player)); init = init)
#         ) 


#     create_critic(critic_dim) = Chain(
#         Dense(critic_dim, netWidth, relu; init = init),
#         [Dense(netWidth, netWidth, relu; init = init) for _ in 1:netLayers]...,
#         Dense(netWidth, 1; init = init),
#         ) 

#     create_policy(player) = DDPGPolicy(
#             behavior_actor = NeuralNetworkApproximator(
#                 model = create_actor(player),
#                 #optimizer = ADAM(),
#                 #optimizer = ADAM(1e-6),
#                 optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-4)),
        
#             ),
#             behavior_critic = NeuralNetworkApproximator(
#                 model = create_critic(critic_dim),
#                 #optimizer = ADAM(),
#                 #optimizer = ADAM(1e-6),
#                 optimizer = Flux.Optimise.Optimiser(ClipNorm(0.5), ADAM(1e-4)),
#             ),
#             target_actor = NeuralNetworkApproximator(
#                 model = create_actor(player),
#             ),
#             target_critic = NeuralNetworkApproximator(
#                 model = create_critic(critic_dim),
#             ),
#             γ = gamma, # discount factor to compute TD error
#             ρ = rho, # controls the percentage of parameters that don't make it to the target: dest .= ρ .* dest .+ (1 - ρ) .* src
#             na = length(action_space(env, player)),

#             start_steps = 200*batchSize, # wait until start_steps before learning
#             start_policy = RandomPolicy(Space([-act_limit..act_limit for _ in 1:length(action_space(env, player))]); rng = rng), # policy that is used until start_steps
#             update_after = 200*batchSize, # wait until update_after before updating

#             act_limit = act_limit, # used to clamp the output of the actor NN, the action vector further passed to a tanh function of unit width with user-specified bounds for each element of the vector
#             act_noise = act_noise, # how much noise to add to the actions
#             rng = rng,
#         )

#     create_trajectory(player) = CircularArraySARTTrajectory(
#             capacity = 1_000_000, # replay buffer capacity
#             state = Vector{Float64} => (length(state(env, player)), ),
#             action = Vector{Float64} => (length(action_space(env, player)), ),
#             reward = Float64 => (1,)
#         )

#     agents = MADDPGManagerAlt(
#         Dict(
#             player => Agent(
#                 policy = NamedPolicy(player, create_policy(player)),
#                 trajectory = create_trajectory(player),
#             ) for player in (:PlusDefect, :MinusDefect)
#             #) for player in (:Speaker, :Listener)
#         ),
#         SARTS, # trace's type
#         batchSize, # batch_size, how many SARTS samples to use in update of target network
#         updateFreq,  # only transfer to target every update_freq steps - I think this overrides by the DDPG update_freq (see line 67 in MADDPG page)
#         0, # initial update_step, set to 0
#         rng
#     )
    
#     if eps_or_hrs == "eps"
#         stop_condition = StopAfterEpisode(nEpisodes, is_show_progress=true)
#         #stop_condition = StopAfterStep(nEpisodes * nSteps, is_show_progress=true)
#     else 
#         stop_condition = StopAfterNSeconds(nEpisodes * 3600.0)
#     end


#     if envMarker == "nem"
#         hook = RLNematicHook(stepStride = stepStride, episodeStride = episodeStride)
#         tagline = "# Nematic defect RL with MADDPG"
#     else 
#         hook = RLToyHook(stepStride = stepStride, episodeStride = episodeStride, 
#         annealBool = annealBool, annealTime = annealTime)
#         tagline = "# Toy spring model RL with MADDPG"
#     end

#     Experiment(agents, env, stop_condition, hook, tagline)
# end



function CreateDDPGExperiment(env; # works for toy or the full RL env
    seed = 0,
    eps_or_hrs = "eps",
    envMarker = "nem", 
    nEpisodes = 10,
    batchSize = 128,
    updateFreq = 100,
    stepStride = 1,
    episodeStride = 5,
    stateTrajStride = 50,
    netLayers = 3,
    netWidth = 64,
    gamma = 0.95f0,
    rho = 0.99f0,
    act_limit = 1.0,
    act_noise = 1e-3,
    annealBool = false,
    annealTime = 100,
    learningRate = 1e-4,
    clipNorm = 0.5
    )
    
    
    rng = env.rng
    A = action_space(env)
    ns = length(state(env))
    na = length(A)

    init = glorot_uniform(rng)

    create_actor() = Chain( 
        Dense(ns, netWidth, relu; init = init),
        [Dense(netWidth, netWidth, relu; init = init) for _ in 1:netLayers]...,
        Dense(netWidth, na; init = init)
        ) 


    create_critic() = Chain(
        Dense(ns + na, netWidth, relu; init = init),
        [Dense(netWidth, netWidth, relu; init = init) for _ in 1:netLayers]...,
        Dense(netWidth, 1; init = init),
        ) 

    agent = Agent(
        policy = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                #optimizer = ADAM(),
                optimizer = Flux.Optimise.Optimiser(ClipNorm(clipNorm), ADAM(learningRate)),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                #optimizer = ADAM(),
                optimizer = Flux.Optimise.Optimiser(ClipNorm(clipNorm), ADAM(learningRate)),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                #optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                #optimizer = ADAM(),
            ),

            γ = gamma, # discount factor to compute TD error
            ρ = rho, # controls the percentage of parameters that don't make it to the target: dest .= ρ .* dest .+ (1 - ρ) .* src
            na = na,
            batch_size = batchSize,

            start_steps = Int(floor((1/2) * 20*batchSize)), # wait until start_steps before learning
            start_policy = RandomPolicy(Space([-act_limit..act_limit for _ in 1:na]); rng = rng), # policy that is used until start_steps
            update_after = Int(floor((1/2) * 20*batchSize)), # wait until update_after before updating
            update_freq = updateFreq,

            act_limit = act_limit, # used to clamp the output of the actor NN, the action vector further passed to a tanh function of unit width with user-specified bounds for each element of the vector
            act_noise = act_noise, # how much noise to add to the actions
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1_000_000,
            state = Vector{Float64} => (ns,),
            action = Vector{Float64} => (na, ),
        ),
    )

    if eps_or_hrs == "eps"
        stop_condition = StopAfterEpisode(nEpisodes, is_show_progress=true)
        #stop_condition = StopAfterStep(nEpisodes * nSteps, is_show_progress=true)
    else 
        stop_condition = StopAfterNSeconds(nEpisodes * 3600.0)
    end


    if envMarker == "nem"
        hook = RLNematicHook(stepStride = stepStride, episodeStride = episodeStride, stateTrajStride = stateTrajStride,
        annealBool = annealBool, annealTime = annealTime)
        tagline = "# Nematic defect RL with DDPG"
    else 
        hook = RLToyHook(stepStride = stepStride, episodeStride = episodeStride, 
        annealBool = annealBool, annealTime = annealTime)
        tagline = "# Toy spring model RL with DDPG"
    end

    Experiment(agent, env, stop_condition, hook, tagline)
end



# function SaveResults(ex, parameters, pathToSave)
#     plusActorModel = deepcopy(ex.policy.agents[:PlusDefect].policy.policy.target_actor.model)
#     minusActorModel = deepcopy(ex.policy.agents[:MinusDefect].policy.policy.target_actor.model)
#     hookResults = deepcopy(ex.hook.episodeList)
#     savedData = (parameters, hookResults, plusActorModel, minusActorModel)
#     jldsave(pathToSave * "SavedData.jld2"; savedData)
# end

function SaveResults(ex, parameters, pathToSave)
    save(pathToSave * "SavedData.jld2", 
    "ex", ex, 
    "parameters", parameters)
end

########################################################################################
#
#                    Initialize and run functions
#
########################################################################################

function InitializeAndRunExperiment(parameters)

    ##########################
    ### Define parameters ###
    ##########################

    ## physics / lattice parameters
    global Nx = parameters["Nx"] # number of grid points
    global Ny = parameters["Ny"]
    global ndt = parameters["ndt"] # number of dt to integrate in one RL step
    global bc = parameters["bc"] # boundary conditions
    global lambdaBE = parameters["lambdaBE"] # flow-alignment coupling parameter
    global A0BE = parameters["A0BE"] # strength of polarization terms in free energy
    global UBE = parameters["UBE"] # sets the equilibrium polarization
    global LBE = parameters["LBE"] # strength of gradient term in free energy
    global GammaBE = parameters["GammaBE"] # rotation diffusion constant
    global friction = parameters["friction"] # subtrate friction term

    global offX = parameters["offX"] # half of initial x separation between defects  
    global offY = parameters["offY"]
    global q = parameters["q"] # overall rotation of nematic field in initial configuration
    global randParam = parameters["randParam"] # overall rotation of nematic field in initial configuration
    global plusTweezerParams = parameters["plusTweezerParams"] 
    global minusTweezerParams = parameters["minusTweezerParams"] 
    global tweezerType = parameters["tweezerType"]

    global fBool = parameters["fBool"]
    global FM = parameters["FM"]

    sP = SimParams(Nx, Ny, ndt, bc, bc, lambdaBE, A0BE, UBE, LBE, GammaBE, friction, [offX, offY, q, randParam], 
        plusTweezerParams, minusTweezerParams, tweezerType, fBool, FM)

    ## parameters of the imposed force law 
    global ks = parameters["ks"] # stiffness times the drag
    global l0 = parameters["l0"] # equilibrium separation
    global kt = parameters["kt"]

    fLP = ForceLawParams(ks, l0, kt)

    ## parameters of the Environment object
    global nSteps = parameters["nSteps"] # number of steps in one episode
#    global nSteps = Int32(floor(3750 / ndt)) # number of steps in one episode
    global bounds = parameters["bounds"] # range of allowed increments of the activityCoefficients
    global taskMarker = parameters["taskMarker"] # label of the task to do
    global rewP = parameters["rewP"] # strength of the penalty used to compute the reward
    global rewT = parameters["rewT"] # penalizes alternate parts of the dynamics
    global updateOncePerStepBool = parameters["updateOncePerStepBool"] # whether to update agents and activityField each step (true) or each dt (false), for efficiency
    global seed = Int(floor(parameters["seed"])) # random seed for network initialization
    rng = StableRNG(seed);

    env = NematicEnv(sP, fLP, rng; nSteps = nSteps, bounds = bounds, taskMarker = taskMarker, rewP = rewP, rewT = rewT, updateOncePerStepBool = updateOncePerStepBool)

    ## parameters of the Experiment object
    global seed = parameters["seed"] # random seed for network initialization
    global eps_or_hrs = parameters["eps_or_hrs"] # terminate after nEpisodes epsiodes ("eps") or nEpisodes hours ("hrs")
    global nEpisodes = parameters["nEpisodes"] # number of episodes / hours 
    global envMarker = "nem"
    global batchSize = Int64(parameters["batchSize"]) # how many samples to include in replay buffer used to train the networks
    global updateFreq = Int64(parameters["updateFreq"]) # how many steps to do before updating the network parameters 
    global stepStride = parameters["stepStride"] # clamp on NN action output
    global episodeStride = parameters["episodeStride"] # 
    global stateTrajStride = parameters["stateTrajStride"] #
    global netLayers = Int32(parameters["netLayers"]) # number of hidden layers in the neural networks 
    global netWidth = Int32(parameters["netWidth"]) # width of hidden layers in the neural networks
    global gamma = Float32(parameters["gamma"]) # discount factor
    global rho = Float32(parameters["rho"]) # soft update factor
    global act_limit = parameters["act_limit"] # clamp on NN action output
    global act_noise = parameters["act_noise"] # action noise scale
    global annealBool = parameters["annealBool"] # whether to anneal the noise in activity
    global annealTime = parameters["annealTime"] # time over which the noise exponentially decay (in units of steps)
    global learningRate = parameters["learningRate"]
    global clipNorm = parameters["clipNorm"]

    ex = CreateDDPGExperiment(env; seed = seed, eps_or_hrs = eps_or_hrs, nEpisodes = nEpisodes, 
        stepStride = stepStride, episodeStride = episodeStride, stateTrajStride = stateTrajStride, envMarker = envMarker,
        batchSize = batchSize, updateFreq = updateFreq, netLayers = netLayers, netWidth = netWidth, 
        gamma = gamma, rho = rho, act_limit = act_limit, act_noise = act_noise, annealBool = annealBool, annealTime = annealTime, 
        learningRate = learningRate, clipNorm = clipNorm)

    ###########################################
    ### Run the experiment and save results ###
    ###########################################

    run(ex)
    
    return ex
end


function RestartExperiment(pathToLoadRestartFile, nEpisodes, eps_or_hrs = "eps")

    ########################################################
    ### Load previous experiment, reset counter, and run ###
    ########################################################
    
    d = load(pathToLoadRestartFile)
    ex = d["ex"]

    if eps_or_hrs == "eps"
        ex.stop_condition = StopAfterEpisode(nEpisodes, is_show_progress=true)
    else 
        ex.stop_condition = StopAfterNSeconds(nEpisodes * 3600.0)
    end

    run(ex)
    
    return ex
end

# function InitializeAndRunToyExperiment(parameters)

#     ##########################
#     ### Define parameters ###
#     ##########################

#     ## physics / lattice parameters
#     global kEl = parameters["kEl"]
#     global pThresh = parameters["pThresh"]
#     global randBool = parameters["randBool"]
#     global pInitArg = parameters["pInitArg"]
#     global ndt = parameters["ndt"]

#     sP = ToySimParams(kEl, pThresh, randBool, pInitArg, ndt)

#     ## parameters of the imposed force law 
#     global kst = parameters["kst"] # stiffness times the drag
#     global pst = parameters["pst"] # equilibrium separation

#     fLP = ToyForceLawParams(kst, pst)

#     ## parameters of the Environment object
#     global nSteps = parameters["nSteps"] # number of steps in one episode
#     global bounds = parameters["bounds"] # range of allowed increments of the activityCoefficients
#     global rewP = parameters["rewP"] # strength of the penalty used to compute the reward
#     global extraActionsBool = parameters["extraActionsBool"] # whether the include y / theta coefficients in action space
#     global seed = parameters["seed"] # random seed for network initialization
    
#     env = ToyEnv(sP, fLP, StableRNG(seed); nSteps = nSteps, bounds = bounds, rewP = rewP, extraActionsBool = extraActionsBool)


#     ## parameters of the Experiment object
    
#     global eps_or_hrs = parameters["eps_or_hrs"] # terminate after nEpisodes epsiodes ("eps") or nEpisodes hours ("hrs")
#     global envMarker = "toy" # "nem" or "toy"
#     global nEpisodes = parameters["nEpisodes"] # number of episodes / hours 
#     global batchSize = parameters["batchSize"] # how many samples to include in replay buffer used to train the networks
#     global updateFreq = parameters["updateFreq"] # how many steps to do before updating the network parameters 
#     global stepStride = parameters["stepStride"] # clamp on NN action output
#     global episodeStride = parameters["episodeStride"] # action noise scale
#     global netLayers = parameters["netLayers"] # number of hidden layers in the neural networks 
#     global netWidth = parameters["netWidth"] # width of hidden layers in the neural networks
#     global gamma = parameters["gamma"] # discount factor
#     global rho = parameters["rho"] # soft update factor
#     global act_limit = parameters["act_limit"] # clamp on NN action output
#     global act_noise = parameters["act_noise"] # action noise scale
#     global annealBool = parameters["annealBool"] # whether to anneal the noise in activity
#     global annealTime = parameters["annealTime"] # time over which the noise exponentially decay (in units of steps)


#     #env = SpeakerListenerEnvCopy(max_steps = 25)

#     # ex = CreateNematicExperiment(env; seed = seed, eps_or_hrs = eps_or_hrs, nEpisodes = nEpisodes, 
#     #     stepStride = stepStride, episodeStride = episodeStride, envMarker = envMarker,
#     #     batchSize = batchSize, updateFreq = updateFreq, netLayers = netLayers, netWidth = netWidth, 
#     #     gamma = gamma, rho = rho, act_limit = act_limit, act_noise = act_noise, annealBool = annealBool, annealTime = annealTime)

    
#     ex = CreateDDPGExperiment(env; seed = seed, eps_or_hrs = eps_or_hrs, nEpisodes = nEpisodes, 
#         stepStride = stepStride, episodeStride = episodeStride, envMarker = envMarker,
#         batchSize = batchSize, updateFreq = updateFreq, netLayers = netLayers, netWidth = netWidth, 
#         gamma = gamma, rho = rho, act_limit = act_limit, act_noise = act_noise, annealBool = annealBool, annealTime = annealTime)


#     # ex = CreateSLExperiment(seed);

#     ###########################################
#     ### Run the experiment and save results ###
#     ###########################################

#     run(ex)
    
#     return ex
# end