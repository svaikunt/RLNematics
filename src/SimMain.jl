module SimMain

    using Random
    using Distributions
    using Interpolations
    using CoherentNoise

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    include("Training.jl")
    include("TestCases.jl")
    include("BerisEdwards.jl")

    export SimParams,
    SimState,
    ForceLawParams,
    InitializeSimState,
    PredictedState,
    UpdateAgentHandlerFromAction!,
    SimStep!

    
    ########################################################################################
    #
    #                    Main simulation functions and structs
    #
    ########################################################################################

    """
    Holds the simulation parameters.
    """
    struct SimParams
        grid::Grid2D
        ndt::Int # how many times to integrate in one RL step

        bcBE_X::String
        bcBE_Y::String 

        beParams::BerisEdwards.BEQParams
        friction::Real 

        initParams::Vector{Real} # use to intialize defect pair, either the values or the bounds of a distribution 

        plusTweezerParams::Dict
        minusTweezerParams::Dict
        tweezerType::String

        fBool::Bool # external forcing function
        FM::Real # strength of forcing

        function SimParams(
            Nx,
            Ny,
            ndt, 
            bcBE_X,
            bcBE_Y,
            lambdaBE,
            A0BE,
            UBE,
            LBE,
            GammaBE,
            friction,
            initParams, 
            plusTweezerParams,
            minusTweezerParams, 
            tweezerType,
            fBool,
            FM)

            return new(Grid2D(Nx, Ny, 1), ndt, bcBE_X, bcBE_Y, BerisEdwards.BEQParams(lambdaBE, A0BE, UBE, LBE, GammaBE), friction, initParams, plusTweezerParams, minusTweezerParams, tweezerType, fBool, FM)
        end
    end


    """
    Holds the fields which make up the simulation state.
    """
    mutable struct SimState
        velocitySoA
        nematicSoA
        activityField
        agentHandler
        lastAgentHandler
        initParams
        externalForcePattern

        function SimState(
            velocitySoA,
            nematicSoA,
            activityField,
            agentHandler,
            lastAgentHandler,
            initParams,
            externalForcePattern
            )

            return new(velocitySoA, nematicSoA, activityField, agentHandler, lastAgentHandler, initParams, externalForcePattern)
        end
    end

    """
    Holds force law parameters.
    """
    struct ForceLawParams
        ks::Real
        l0::Real 
        kt::Real 
        
        function ForceLawParams(ks, l0, kt = 0.0)
            return new(ks, l0, kt)
        end 
    end 


    """
        InitializeSimState(sP)
    
    
    Initializes the simulation state from `sP`.

    Return [`SimState`](@ref).
    """
    function InitializeSimState(sP, rng, fLP, taskMarker, nSteps)

        ## initialize the nematic field
        qInit = (1/4) + (3/4) * sqrt(1 - 8 / (3 * sP.beParams.U)) # set initial magnitude to the equilibrium value

        # initParams is [offX, offY, q / randFacq, randFac]
        offXI = sP.initParams[1]
        offYI = sP.initParams[2]
        rP = sP.initParams[4]

        if (taskMarker == "sepMin") || (taskMarker == "sepPlus") || (taskMarker == "orPlus") || (taskMarker == "orPlusPert")
            offX = rand(rng, Uniform((1-rP) * offXI, (1+rP) * offXI))
            if (taskMarker == "sepMin") || (taskMarker == "sepPlus")
                offY = 0
                q = 0 + sP.initParams[3]
            else
                offY = rand(rng, Uniform(-offYI, offYI))
                q = rand(rng, Uniform(-sP.initParams[3], sP.initParams[3]))
#		q = rand([-1, 1]) * rand(rng, Uniform(0.1, sP.initParams[3]))
            end
            ips = [offX, offY, q]
            nematicSoA = Training.InitializeDefectPair(sP.grid, qInit, ips...)

        elseif taskMarker == "orbit"
            offX = rand(rng, Uniform((1-rP) * offXI, (1+rP) * offXI))
            offQ = 3*pi/4
            q = rand(rng, Uniform(offQ - sP.initParams[3], offQ + sP.initParams[3]))
            offY = offYI
            ips = [offX, offY, q]
            nematicSoA = Training.InitializeDefectPair(sP.grid, qInit, ips...)

            # translate so the minus defect is centered
            xObservables = 0:1:(sP.grid.Nx+1)
            yObservables = 0:1:(sP.grid.Ny+1)
            obs = (xObservables, yObservables)
            XXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XXValues), Gridded(Linear()))
            XYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XYValues), Gridded(Linear()))
            YXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YXValues), Gridded(Linear()))
            YYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YYValues), Gridded(Linear()))
            newNematicSoA = deepcopy(nematicSoA)
            for i=1:sP.grid.Nx, j=1:sP.grid.Ny
                newI = mod1(i - offX, sP.grid.Nx)
                newNematicSoA.XXValues[i,j] = XXFunc(newI,j)
                newNematicSoA.XYValues[i,j] = XYFunc(newI,j)
                newNematicSoA.YXValues[i,j] = YXFunc(newI,j)
                newNematicSoA.YYValues[i,j] = YYFunc(newI,j)
            end
            nematicSoA = newNematicSoA

        elseif taskMarker == "follower"
            offX = rand(rng, Uniform((1-rP) * offXI, (1+rP) * offXI))
            offY = rand(rng, Uniform(-offYI, offYI))
            offQ = pi/2
            q = rand(rng, Uniform(offQ - sP.initParams[3], offQ + sP.initParams[3]))
            ips = [offX, offY, q]
            nematicSoA = Training.InitializeDefectPair(sP.grid, qInit, ips...)

            # translate so the plus defect is centered
            xObservables = 0:1:(sP.grid.Nx+1)
            yObservables = 0:1:(sP.grid.Ny+1)
            obs = (xObservables, yObservables)
            XXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XXValues), Gridded(Linear()))
            XYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XYValues), Gridded(Linear()))
            YXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YXValues), Gridded(Linear()))
            YYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YYValues), Gridded(Linear()))
            newNematicSoA = deepcopy(nematicSoA)
            for i=1:sP.grid.Nx, j=1:sP.grid.Ny
                newI = mod1(i + offX, sP.grid.Nx)
                newNematicSoA.XXValues[i,j] = XXFunc(newI,j)
                newNematicSoA.XYValues[i,j] = XYFunc(newI,j)
                newNematicSoA.YXValues[i,j] = YXFunc(newI,j)
                newNematicSoA.YYValues[i,j] = YYFunc(newI,j)
            end
            nematicSoA = newNematicSoA
        end

        ## initialize the other fields
        agentHandler = Training.AgentHandler(sP.grid, nematicSoA, sP.bcBE_X, sP.bcBE_Y, sP.tweezerType)
        lastAgentHandler = deepcopy(agentHandler)
        activityField = ScalarSoA2D(sP.grid)
        velocitySoA = VectorSoA2D(sP.grid)

        ## set the external forcing function
        retVVM = TestCases.TaylorGreenVortexT2DTiled(sP.grid, 2, 2)
        vM = VectorMesh2D(sP.grid)
        vM.Values .= retVVM
        TGForcePatternSoA = ConvertVectorMeshToSoA2D(sP.grid, vM)
        sampler = simplex_1d()
        newsampler = CoherentNoise.scale(sampler, 100)
        vals = sP.FM .* [CoherentNoise.sample(newsampler, t) for t in 1:(2 * sP.ndt * nSteps)]
        timeSchedule = Training.ExternalForceTimeSchedule(vals)
        externalForcePattern = Training.ExternalForce2D(TGForcePatternSoA, timeSchedule); # combine timing and spatial

        ## create the sS object
        sS = SimState(velocitySoA, nematicSoA, activityField, agentHandler, lastAgentHandler, ips, externalForcePattern)
    
        ## run short eq steps to smooth pbc edges and reset agentHandler members
        SimStep!(sS, sP, 100, false)

        sS.agentHandler = Training.AgentHandler(sP.grid, sS.nematicSoA, sP.bcBE_X, sP.bcBE_Y, sP.tweezerType)

        ## set the default tweezerParams to which are specified in sP.plusTweezerParams, otherwise they'll be 0
        for (key, val) in sP.plusTweezerParams
            if sP.tweezerType == "quad"
                if key == "cutoff"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cutoff = val 
                elseif key == "width" 
                    sS.agentHandler.PlusDefects[1].activityCoefficients.width = val
                elseif key == "c0"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = val
                elseif key == "cx"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cx = val
                elseif key == "cy"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cy = val
                elseif key == "cxx"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cxx = val
                elseif key == "cyy"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cyy = val
                elseif key == "theta"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.theta = val
                end 
            else 
                if key == "cutoff"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.cutoff = val 
                elseif key == "width" 
                    sS.agentHandler.PlusDefects[1].activityCoefficients.width = val
                elseif key == "c0"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.c0 = val
                elseif key == "m"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.m = val
                elseif key == "theta"
                    sS.agentHandler.PlusDefects[1].activityCoefficients.theta = val
                end
            end
        end 
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0x = sS.agentHandler.PlusDefects[1].Position[1]
        sS.agentHandler.PlusDefects[1].activityCoefficients.r0y = sS.agentHandler.PlusDefects[1].Position[2]

        for (key, val) in sP.minusTweezerParams
            if sP.tweezerType == "quad"
                if key == "cutoff"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cutoff = val 
                elseif key == "width" 
                    sS.agentHandler.MinusDefects[1].activityCoefficients.width = val
                elseif key == "c0"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.c0 = val
                elseif key == "cx"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cx = val
                elseif key == "cy"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cy = val
                elseif key == "cxx"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cxx = val
                elseif key == "cyy"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cyy = val
                elseif key == "theta"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.theta = val
                end 
            else
                if key == "cutoff"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.cutoff = val 
                elseif key == "width" 
                    sS.agentHandler.MinusDefects[1].activityCoefficients.width = val
                elseif key == "c0"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.c0 = val
                elseif key == "m"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.m = val
                elseif key == "theta"
                    sS.agentHandler.MinusDefects[1].activityCoefficients.theta = val
                end  
            end 
        end 
        sS.agentHandler.MinusDefects[1].activityCoefficients.r0x = sS.agentHandler.MinusDefects[1].Position[1]
        sS.agentHandler.MinusDefects[1].activityCoefficients.r0y = sS.agentHandler.MinusDefects[1].Position[2]
        

   
        sS.lastAgentHandler = deepcopy(sS.agentHandler)

        return sS
    end 

    @inline TanhClamp(x::Real, width::Real, cutoff::Real) = 0.5 * (1.0 -tanh((x - cutoff) / width))
    

    """
        SimStep!(sS, sP)

    Integrates for `ndt` and update the fields in `sS`.
    """
    function SimStep!(sS, sP, initEq = 0, updateOncePerStepBool = false) # integrates for ndt and updates the fields in sS

        sS.lastAgentHandler = deepcopy(sS.agentHandler) # copy previous defect positions to integrate the force law for the reward

        if initEq == 0
            ndt = sP.ndt
        else 
            ndt = initEq
        end

        if updateOncePerStepBool
            Training.SetActivityFieldFromAgents!(sP.grid, sS.activityField, sS.agentHandler, sP.bcBE_X, sP.bcBE_Y, false)
        end

        for t in 1:ndt
            
            ### Compute new velocity field but don't set it yet
            eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(sP.grid, sS.nematicSoA, sP.beParams, sP.bcBE_X, sP.bcBE_Y)
            aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(sS.nematicSoA, sP.grid, sS.activityField)
            polymerForceSoA = DivTensorOnSoA2D(sP.grid, MatrixTransposeOnSoA2D(sP.grid, eSoA), BCDerivDict[sP.bcBE_X], BCDerivDict[sP.bcBE_Y])
            activeForceSoA = DivTensorOnSoA2D(sP.grid, MatrixTransposeOnSoA2D(sP.grid, aSoA), BCDerivDict[sP.bcBE_X], BCDerivDict[sP.bcBE_Y])
            AddVectorSoA2D!(polymerForceSoA, activeForceSoA)

            if sP.fBool
                externalForceSoA = MultiplyVectorSoA2D(sP.grid, sS.externalForcePattern.SpatialPattern, popfirst!(sS.externalForcePattern.TimeSchedule.Values))
                AddVectorSoA2D!(polymerForceSoA, externalForceSoA)
            end

            MultiplyVectorSoA2D!(polymerForceSoA, 1 / sP.friction)

            ### Update the fields in place 

            if (!updateOncePerStepBool)
                # activityField
                Training.SetActivityFieldFromAgents!(sP.grid, sS.activityField, sS.agentHandler, sP.bcBE_X, sP.bcBE_Y, false)

                # agentHandler
                Training.UpdateAgentHandler!(sP.grid, sS.nematicSoA, sS.agentHandler, sP.bcBE_X, sP.bcBE_Y, sP.tweezerType)
            end

            
            # nematicSoA
            (OmegaSoA, PsiSoA) = OmegaPsi2DSoA(sP.grid, sS.velocitySoA, sP.bcBE_X, sP.bcBE_Y)
            BerisEdwards.PredictorCorrectorStepQ2DSoA!(sP.grid, OmegaSoA, PsiSoA, sS.velocitySoA, sS.nematicSoA, 1, sP.beParams, sP.bcBE_X, sP.bcBE_Y)

            # velocitySoA
            SetVectorFromSoA2D!(sS.velocitySoA, polymerForceSoA) 

        end 

        if updateOncePerStepBool
            Training.UpdateAgentHandler!(sP.grid, sS.nematicSoA, sS.agentHandler, sP.bcBE_X, sP.bcBE_Y, sP.tweezerType)
        end
    end

end # Main
