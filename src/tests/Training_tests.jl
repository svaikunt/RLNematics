using Test

include("../Training.jl")
using .Training

### Defect Agents

myActivityCoefficients = Training.ActivityCoefficients()

myDefectAgent = Training.DefectAgent(false, "test_label", [3,2], 0.5)

# Agent Handler constructor test
# AgentHandler(grid, nematicSoA, bcx, bcy)
# Grid2D, TensorSoA2D (SharedStructs.jl), String, String

# GetMinimumDistance

# GetNematicDivergenceInterp

# UpdateAgentHandler

# UpdateActivityCoefficientsPosition

# UpdateActivityCoefficients

# BasisCoefficients

# ActivityFromCoefficients

# SetActivityFieldFromAgents

### Pulling Protocol

# AddkLeg

# evalkLeg

# evalk

# AddrExtLeg

# evalrExtLeg

# evalrExt

### Initializaton Functions

# InitializeDefectPair

# InitializeDefect

# InitializeWave

# GenerateTrajectoryX
