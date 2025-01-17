#!/bin/bash
#SBATCH --job-name=nematic
#SBATCH --output=training_batch.out
#SBATCH --error=training_batch.err
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=48000

module load julia
julia /project/svaikunt/csfloyd/RLNematic/JuliaCode/SimulationInputs/ForceLawRL.jl inputFile outputDir

