include( "SharedStructs.jl")
using .SharedStructs
include("MathFunctions.jl")
using .MathFunctions
include("TestCases.jl")
include("Misc.jl")
include("Analysis.jl")
include("RLEnvironment.jl")
include("RLExperiment.jl")
#using JLD
#using JLD2
using FileIO
using Statistics
using BenchmarkTools
using ReinforcementLearning

annealTimeVec = [1e4, 1e5, 1e6, 1e7]
gammaVec = [0.0, 0.5, 0.99]
rhoVec = [0.9, 0.95, 0.995, 0.9995]
netLayersVec = [1, 2, 3]
netWidthVec = [32, 64]
learningRateVec = [1e-6, 1e-5, 1e-4, 1e-3]
batchSizeVec = [32, 64, 128, 256]
seedVec = [1,2,3,4,5,6,7,8,9,10,11,12,13]
#seedVec = [10, 20, 30, 40, 50]
ksVec = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4][1:2:end]
ksVec = [0e-4, 0.5e-4, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4, 8e-4]

dir = ""
subDir = "Dirs_seed_ks_rel_4"
objName = "policy_seed_ks_rel_4_full"
case = "dV"
v1 = "seed"
v1Vec = seedVec
v2 = "ks"
v2Vec = ksVec

#func = x -> Analysis.GetRewardTrajectory(x)
#func = x -> Analysis.GetPlusOrientationTrajectory(x)
func = x -> Analysis.GetBehaviorPolicy(x)

if case == "sV"
    ## single variable loop
    pathBase = "/project/svaikunt/csfloyd/RLNematic/Dirs/" * dir * "/" * subDir
    bigV1Dict = Dict([])
    @time for v1i in v1Vec
        try
            pathName = pathBase * "/" * v1 * "_" * string(v1i) * "/"
            d = load(pathName * "SavedData.jld2")
            ret = func(d)
            bigV1Dict[v1i] = ret
        catch e
            println("Error reading " * string(v1i))
	    showerror(stdout, e)
        end
    end
    global retDict = bigV1Dict
end

if case == "dV"
    ## double variable loop
    pathBase = "/project/svaikunt/csfloyd/RLNematic/Dirs/" * dir * "/" * subDir
    bigV1Dict = Dict([])
    @time for v1i in v1Vec
        bigV2Dict = Dict([])
        for v2i in v2Vec
            try
                pathName = pathBase * "/" * v1 * "_" * string(v1i) * "/" * v2 * "_" * string(v2i) * "/"
                d = load(pathName * "SavedData.jld2")
                ret = func(d)
                bigV2Dict[v2i] = ret
            catch e
                println("Error reading " * string(v1i) * "_" * string(v2i))
		        showerror(stdout, e)
            end                
        end
        bigV1Dict[v1i] = bigV2Dict
    end
    global retDict = bigV1Dict
end


# save results
outputDir = "/project/svaikunt/csfloyd/RLNematic/Dirs/" * dir * "/AnalyzedData/"
pathName = outputDir * objName * ".jld2"
save(pathName, "retDict", retDict)
