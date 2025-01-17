#ksVec = [0.1, 0.5]
ksVec = [1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3]
ksVec = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4][1:2:end]
ksVec = [0e-4, 0.5e-4, 1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4, 5e-4, 5.5e-4, 6e-4, 6.5e-4, 7e-4, 7.5e-4, 8e-4]
cutoffVec = [5, 7.5, 10]
ndtVec = [5, 10, 25, 50]
l0Vec = [40, 45, 50]

ktVec = 0.025 .* [0.5, 1, 1.5, 2]
rewTVec = [0.0, 0.5, 1]

rewTVec = [10, 20, 30, 40, 50]

seedVec = [10, 20, 30, 40, 50]
#seedVec = [1,2,3,4,5,6,7,8,9,10]
seedVec = [11, 12, 13]
#seedVec = [1,2,3,4,5]
FMVec = [0.0, 1e-2]
annealTimeVec = [1e4, 1e5, 1e6, 1e7]
gammaVec = [0.0, 0.5, 0.99]
#gammaVec = [0.5, 0.99]
rhoVec = [0.9, 0.95, 0.995, 0.9995]
netLayersVec = [1, 2, 3]
netWidthVec = [32, 64]
learningRateVec = [1e-6, 1e-5, 1e-4, 1e-3]
batchSizeVec = [32, 64, 128, 256]
rewPVec = [0.1, 1, 10, 100]
ndtVec = [10, 50, 100, 200]
clipNormVec = [0.25, 0.5, 0.75, 1]
updateFreqVec = [1, 5, 10, 50]

trialParamDict = Dict(
"ks" => ksVec,
#"updateFreq" => updateFreqVec
#"kt" => ktVec
#"rewT" => rewTVec
"seed" => seedVec,
#"annealTime" => annealTimeVec
#"gamma" => gammaVec
#"rho" => rhoVec
#"netLayers" => netLayersVec,
#"netWidth" => netWidthVec
#"learningRate" => learningRateVec
#"batchSize" => batchSizeVec
#"rewP" => rewPVec
#"ndt" => ndtVec
#"clipNorm" => clipNormVec
#"cutoff" => cutoffVec
#"ndt" => ndtVec
#"l0" => l0Vec
#"FM" => FMVec
)

global baseDir = pwd() * "/";
global Dirs = baseDir * "/Dirs/Dirs_seed_ks_rel_4_ex/"

function ReplaceBatchRunDelete(batchFile, inputFile, outputDir)
    f = open(batchFile)
    (tmppath, tmpio) = mktemp()
    try
        lines = readlines(f)
        for l in lines
            sl = split(l)
            if (length(sl) > 0) && (sl[1] == "julia")
                ns = replace(l, "inputFile" => inputFile)
                ns = replace(ns, "outputDir" => outputDir)
                write(tmpio, ns)
                write(tmpio, "\n")

            else
                write(tmpio, l)
                write(tmpio, "\n")

            end
        end
    finally
        close(f)
        close(tmpio)
    end
    newBatch = outputDir * "slatboltz.sh"
    mv(tmppath, newBatch, force = true)
    oldDir = pwd()
    cd(outputDir)
    try
        run(`sbatch slatboltz.sh`)
    catch
        println("Failed to submit the batch job.")
    end
    cd(oldDir)
    rm(newBatch, force = true)
end

function AddLineToFile!(inputFile, newLine)
    f = open(inputFile, "a")
    write(f, newLine)
    write(f, "\n")
    close(f)
end


function MakeDirectoriesAndRun(currParamDict, currDir, currInputFile)

    global baseDir

    currParam = collect(keys(currParamDict))[1]

    if length(collect(keys(currParamDict))) == 1 # reached the bottom
        for val in currParamDict[currParam]
            # create the new directory
            newDir = currDir * "$currParam"*"_"*"$val/"
            mkdir(newDir)
            # update the input file
            newInputFile = newDir * "inputFile.txt"
            newLine = string(currParam) * "     " * string(val)
            cp(currInputFile, newInputFile)
            AddLineToFile!(newInputFile, newLine)
            cp(baseDir * "latboltz.sh", newDir * "latboltz.sh", force = true)
            ReplaceBatchRunDelete(newDir * "latboltz.sh", newInputFile, newDir)
            rm(newDir * "latboltz.sh")
        end
        return
    end

    newParamDict = deepcopy(currParamDict)
    delete!(newParamDict, currParam)

    for val in currParamDict[currParam]
        # create the new directory
        newDir = currDir * "$currParam"*"_"*"$val/"
        mkdir(newDir)
        # update the input file
        newInputFile = newDir * "tempInputFile.txt"
        newLine = string(currParam) * "     " * string(val)
        cp(currInputFile, newInputFile)
        AddLineToFile!(newInputFile, newLine)
        # do the recursive call
        MakeDirectoriesAndRun(newParamDict, newDir, newInputFile)
        # delete the temporary input file
        rm(newDir * "tempInputFile.txt")
    end
end


# make sure the directory is empty
try
    rm(Dirs, recursive = true)
catch
end
mkdir(Dirs)

# do it all
MakeDirectoriesAndRun(trialParamDict, Dirs, baseDir * "/baseInput.txt")
