srcPath = joinpath(@__DIR__, "../")
include(srcPath * "RLEnvironment.jl")
include(srcPath * "RLExperiment.jl")
include(srcPath * "Misc.jl")

args = Base.ARGS;

# set input/output
inputFile = string(args[1])
outputDir = string(args[2])

# set default parameters
defaultParams = Dict(
## physics / lattice parameters
"Nx" => 100, # number of grid points
"Ny" => 100,
"ndt" => 50, # number of dt to integrate in one RL step
"bc" => "pbc", # boundary conditions
"lambdaBE" => 0.7, # flow-alignment coupling parameter
"A0BE" => 0.1, # strength of polarization terms in free energy
"UBE" => 3.5, # sets the equilibrium polarization
"LBE" => 0.1, # strength of gradient term in free energy
"GammaBE" => 1.0, # rotation diffusion constant
"friction" => 10, # subtrate friction term

"offX" =>  25, # half of initial x separation between defects  
"offY" => 1e-4, # interpreted as a range around 0 if > 0
"q" => 1e-4, # overall rotation of nematic field in initial configuration
"randParam" => 0.25,

"tweezerType" => "quad",

"plusTweezerParams" => Dict(
        "c0" => 0,
        "cxx" => 0,
        "cyy" => 0,
        "width" => 5.0,
        "cutoff" => 1.0
        ), 
        
"minusTweezerParams" => Dict(
        "c0" => 7.5,
        "cxx" => 0,
        "cyy" => 0,
        "width" => 5.0,
        "cutoff" => 1.0
        ),

## parameters of the imposed force law 
#"ks" => 0.15, # stiffness times the drag
"ks" => 0.0005, # stiffness times the drag
"l0" => 50.0, # equilibrium separation
"kt" => 0.025,

# external force   
"fBool" => false,
"FM" => 0e0,

## parameters of the Environment object
"nSteps" => 75, #150, # number of steps in one episode
"bounds" => [10, 20], #[offset, c0Range] range of allowed increments of the activityCoefficients
"taskMarker" => "sepMin",
"rewP" => 1e1, # strength of the penalty used to compute the reward
"rewT" => 5,
"updateOncePerStepBool" => true, # whether to update agents and activityField each step (true) or each dt (false), for efficiency
    
## parameters of the Experiment object
"seed" => 20, # random seed for network initialization
"eps_or_hrs" => "hrs", # terminate after nEpisodes epsiodes ("eps") or nEpisodes hours ("hrs")
"envMarker" => "nem",
"nEpisodes" => 3, # number of episodes / hours
"stepStride" => 1,
"episodeStride" => 1,
"stateTrajStride" => 2*25,
"batchSize" => 32, #128, # how many samples to include in replay buffer used to train the networks
"updateFreq" => 5, # how many steps to do before updating the network parameters 
"netLayers" => 2,
"netWidth" => 32,
"gamma" => 0.99f0, 
"rho" => 0.995f0,
"learningRate" => 1e-3,
"clipNorm" => 1.0,
    
"act_limit" => 1.0,
"act_noise" => 0.05,
"annealBool" => false,
"annealTime" => 1e4,


## restart parameters
"restartBool" => false, # whether to load experiment from a previous SavedData file or not 
"parentDirectoryName" => "Dirs_follower_rewT", # the name of the parent folder containing all the param trials
"restartLabel" => "_R" # the label which has been appended to parentDirectoryName directory manually by the user
);

# set parameters
inputParams = Misc.ParseInput(inputFile);
parameters = deepcopy(defaultParams)
for p in keys(inputParams)
    parameters[p] = inputParams[p]
end

println("Starting experiment.")
if parameters["restartBool"]

    # the next 2 lines convert, e.g., ".../Dirs/Dirs_ks/ks_1/" to ".../Dirs/Dirs_ks_R/ks_1/SavedData.jld2"
    labeledOutputDir = replace(outputDir, parameters["parentDirectoryName"] => parameters["parentDirectoryName"] * parameters["restartLabel"]) 
    pathToLoadRestartFile = labeledOutputDir * "SavedData.jld2" 

    @time ex = RestartExperiment(pathToLoadRestartFile, parameters["nEpisodes"], parameters["eps_or_hrs"]);
else
    @time ex = InitializeAndRunExperiment(parameters);
end

println(outputDir);
SaveResults(ex, parameters, outputDir);

println("Done saving results.")
