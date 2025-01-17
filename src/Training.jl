module Training

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions

    using Interpolations
 
    include("BerisEdwards.jl")
    include("Analysis.jl")
    include("Misc.jl")

    ########################################################################################
    #
    #                               Defect agents
    #
    ########################################################################################

    abstract type ActivityCoefficients end

    mutable struct ActivityCoefficientsQuad <: ActivityCoefficients
        r0x::Real
        r0y::Real
        c0::Real
        cx::Real
        cy::Real 
        cxx::Real 
        cyy::Real
        theta::Real 
        cutoff::Real
        width::Real
        label::String
        function ActivityCoefficientsQuad() 
            return new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "quad")
        end
    end

    mutable struct ActivityCoefficientsSin <: ActivityCoefficients
        r0x::Real
        r0y::Real
        c0::Real
        m::Real
        theta::Real 
        cutoff::Real
        width::Real
        label::String
        function ActivityCoefficientsSin() 
            return new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "sin")
        end
    end

    mutable struct DefectAgent
        Sign::Bool # true = plus half, false = minus half
        Label::String
        Position::Array{Real}
        Orientation::Real # angle of polarization
        tweezerType::String
        activityCoefficients::ActivityCoefficients
        function DefectAgent(Sign, Label, Position, Orientation, tweezerType = "quad")
            if tweezerType == "quad"
                return new(Sign, Label, Position, Orientation, "quad", ActivityCoefficientsQuad()) 
            elseif tweezerType == "sin"
                return new(Sign, Label, Position, Orientation, "sin", ActivityCoefficientsSin()) 
            end
        end
    end

    mutable struct AgentHandler
        PlusDefects::AbstractVector
        MinusDefects::AbstractVector
        PlusCount::Int # use to give unique ID's
        MinusCount::Int # use to give unique ID's
        tweezerType::String 
 
        function AgentHandler(grid, nematicSoA, bcx, bcy, tweezerType = "quad")

            (XFunc, YFunc) = GetNematicDivergenceInterp(grid, nematicSoA, bcx, bcy)
            (XFuncM, YFuncM) = GetNematicDivergenceInterp(grid, MirrorNematic(nematicSoA), bcx, bcy)

            (xsp, ysp, xsm, ysm) = Analysis.GetNematicDefects(grid, nematicSoA)
            numP = length(xsp)
            numM = length(xsm)

            PlusDefects = []
            for n = 1:numP 
                ex = XFunc(xsp[n], ysp[n])
                ey = YFunc(xsp[n], ysp[n])
                angle = Analysis.thetaFromPVec([ex,ey]) 
                push!(PlusDefects, DefectAgent(true, "P"*string(n), [xsp[n], ysp[n]], angle, tweezerType))
            end 

            MinusDefects = []
            for n = 1:numM 
                ex = XFuncM(xsm[n], ysm[n])
                ey = YFuncM(xsm[n], ysm[n])
                angle =  mod(- Analysis.thetaFromPVec([ex,ey]) / 3, 2*pi/3)
                push!(MinusDefects, DefectAgent(false, "M"*string(n), [xsm[n], ysm[n]], angle, tweezerType))
            end 

            return new(PlusDefects, MinusDefects, length(PlusDefects), length(MinusDefects), tweezerType)
        end
    end

    function GetMinimumDistance(posA, posList)
        distList = [VectorNorm2D(posList[t] .- posA) for t in 1:length(posList)]
        return findmin(distList)[2] # returns index of posList closest to posA
    end

    function MirrorNematic(nematicSoA)

        newNematicSoA = deepcopy(nematicSoA)
        newNematicSoA.XYValues .*= -1.0
        newNematicSoA.YXValues .*= -1.0
        return newNematicSoA
    end

    function GetNematicDivergenceInterp(grid, nematicSoA, bcx, bcy)

        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]

        divQSoA = DivTensorOnSoA2D(grid, nematicSoA, bcDerivX, bcDerivY)
        NormalizeVectorSoA!(divQSoA)

        # interpolate the divergence field 
        xObservables = 0:1:(grid.Nx+1)
        yObservables = 0:1:(grid.Ny+1)
        obs = (xObservables, yObservables)
        XFunc = interpolate(obs, ExpandForInterpolation(divQSoA.XValues), Gridded(Linear()))
        YFunc = interpolate(obs, ExpandForInterpolation(divQSoA.YValues), Gridded(Linear()))

        return (XFunc, YFunc)
    end

    function UpdateAgentHandler!(grid, nematicSoA, agentHandler, bcx, bcy, tweezerType)

        (xsp, ysp, xsm, ysm) = Analysis.GetNematicDefects(grid, nematicSoA)
        numP = length(xsp)
        numM = length(xsm)

        newPlusPosList = [[xsp[n], ysp[n]] for n in 1:length(xsp)]
        newMinusPosList = [[xsm[n], ysm[n]] for n in 1:length(xsm)]

        plusPosList = [agentHandler.PlusDefects[n].Position for n in 1:length(agentHandler.PlusDefects)]
        minusPosList = [agentHandler.MinusDefects[n].Position for n in 1:length(agentHandler.MinusDefects)]

        ### assign positions
        if numP == length(plusPosList) # update the positive defects
            for n in 1:numP 
                pos = newPlusPosList[n]
                ind = GetMinimumDistance(pos, plusPosList)
                agentHandler.PlusDefects[ind].Position .= pos 
            end 
        
        elseif numP < length(plusPosList) # delete plus defects from the agentHandler
            visitedInds = []
            for n in 1:numP 
                pos = newPlusPosList[n]
                ind = GetMinimumDistance(pos, plusPosList)
                agentHandler.PlusDefects[ind].Position .= pos 
                push!(visitedInds, ind)
            end 
            removeInds = []
            for n in 1:length(agentHandler.PlusDefects)
                if !(n in visitedInds)
                    push!(removeInds, n)
                end 
            end 
            deleteat!(agentHandler.PlusDefects, removeInds)

        else # add plus defects to the agentHandler
            visitedInds = []
            for n in 1:length(plusPosList)
                pos = plusPosList[n] 
                ind = GetMinimumDistance(pos, newPlusPosList)
                agentHandler.PlusDefects[n].Position .= newPlusPosList[n] 
                push!(visitedInds, ind)
            end
            for n in 1:length(newPlusPosList)
                if !(n in visitedInds)
                    agentHandler.PlusCount += 1
                    # initialize with 0.0 angle - will be updated later
                    push!(agentHandler.PlusDefects, DefectAgent(true, "P"*string(agentHandler.PlusCount), newPlusPosList[n], 0.0, tweezerType))
                end 
            end
        end

        if numM == length(minusPosList) # update the minus defects
            for n in 1:numM 
                pos = newMinusPosList[n]
                ind = GetMinimumDistance(pos, minusPosList)
                agentHandler.MinusDefects[ind].Position .= pos 
            end 

        elseif numM < length(minusPosList) # delete minus defects from the agentHandler
            visitedInds = []
            for n in 1:numM 
                pos = newMinusPosList[n]
                ind = GetMinimumDistance(pos, minusPosList)
                agentHandler.MinusDefects[ind].Position .= pos 
                push!(visitedInds, ind)
            end 
            removeInds = []
            for n in 1:length(agentHandler.MinusDefects)
                if !(n in visitedInds)
                    push!(removeInds, n)
                end 
            end 
            deleteat!(agentHandler.MinusDefects, removeInds)
        else # add minus defects to the agentHandler
            visitedInds = []
            for n in 1:length(minusPosList)
                pos = minusPosList[n] 
                ind = GetMinimumDistance(pos, newMinusPosList)
                agentHandler.MinusDefects[n].Position .= newMinusPosList[n] 
                push!(visitedInds, ind)
            end
            for n in 1:length(newMinusPosList)
                if !(n in visitedInds)
                    agentHandler.MinusCount += 1
                    # initialize with 0.0 angle - will be updated later
                    push!(agentHandler.MinusDefects, DefectAgent(false, "M"*string(agentHandler.MinusCount), newMinusPosList[n], 0.0, tweezerType))
                end 
            end
        end

        ### assign orientations
        (XFunc, YFunc) = GetNematicDivergenceInterp(grid, nematicSoA, bcx, bcy)
        (XFuncM, YFuncM) = GetNematicDivergenceInterp(grid, MirrorNematic(nematicSoA), bcx, bcy)

        for n in 1:length(agentHandler.PlusDefects)
            ex = XFunc(agentHandler.PlusDefects[n].Position...)
            ey = YFunc(agentHandler.PlusDefects[n].Position...)
            angle = Analysis.thetaFromPVec([ex,ey]) 
            agentHandler.PlusDefects[n].Orientation = angle 
        end

        for n in 1:length(agentHandler.MinusDefects)
            ex = XFuncM(agentHandler.MinusDefects[n].Position...)
            ey = YFuncM(agentHandler.MinusDefects[n].Position...)
            angle =  mod(- Analysis.thetaFromPVec([ex,ey]) / 3, 2*pi/3)
            agentHandler.MinusDefects[n].Orientation = angle 
        end

    end 

    function UpdateActivityCoefficientsPosition!(grid, parameters, activityCoefficients, r0x, r0y)
        if parameters["bcBE_X"] == "pbc"
            activityCoefficients.r0x = mod1(r0x, grid.Nx)
        else 
            activityCoefficients.r0x = r0x 
        end
        if parameters["bcBE_Y"] == "pbc"
            activityCoefficients.r0y = mod1(r0y, grid.Ny)
        else 
            activityCoefficients.r0y = r0y
        end
    end

    function UpdateActivityCoefficients!(activityCoefficients, args...)
        if activityCoefficients.label == "quad"
            UpdateActivityCoefficientsQuad!(activityCoefficients, args...)
        elseif activityCoefficients.label == "sin"
            UpdateActivityCoefficientsSin!(activityCoefficients, args...)
        end
    end
            
    function UpdateActivityCoefficientsQuad!(activityCoefficients, c0, cx, cy, cxx, cyy, theta, cutoff, width)
        activityCoefficients.c0 = c0
        activityCoefficients.cx = cx
        activityCoefficients.cy = cy
        activityCoefficients.cxx = cxx
        activityCoefficients.cyy = cyy
        activityCoefficients.theta = theta
        activityCoefficients.cutoff = cutoff
        activityCoefficients.width = width
    end

    function UpdateActivityCoefficientsSin!(activityCoefficients, c0, m, theta, cutoff, width)
        activityCoefficients.c0 = c0
        activityCoefficients.m = m
        activityCoefficients.theta = theta
        activityCoefficients.cutoff = cutoff
        activityCoefficients.width = width
    end
    function BasisCoefficients(activityCoefficients) 
        c = cos(activityCoefficients.theta)
        s = sin(activityCoefficients.theta)
        cxN = activityCoefficients.cx * c + activityCoefficients.cy * s
        cyN = activityCoefficients.cy * c - activityCoefficients.cx * s
        cxxN = activityCoefficients.cxx * c^2 + activityCoefficients.cyy * s^2 
        cyyN = activityCoefficients.cxx * s^2 + activityCoefficients.cyy * c^2
        cxyN = 2 * c * s * (activityCoefficients.cyy - activityCoefficients.cxx)

        return (cxN, cyN, cxxN, cyyN, cxyN)
    end 

    @inline TanhBump(x::Real, width::Real, cutoff::Real) = 0.5 * (1.0 -tanh((x - cutoff) / width))

    function ActivityFromCoefficientsQuad(grid, offSets, xOffs, yOffs, activityCoefficients, cxN, cyN, cxxN, cyyN, cxyN, tFac = 5)
        rMag = sqrt.(xOffs.^2 .+ yOffs.^2)
        cutOffFac = zeros(size(rMag))
        thresh =  activityCoefficients.cutoff + tFac * activityCoefficients.width
        width = activityCoefficients.width
        cutoff = activityCoefficients.cutoff
        @inline for j in 1:grid.Ny, i in 1:grid.Nx
            if rMag[i,j] < thresh
                cutOffFac[i,j] = TanhBump(rMag[i,j], width, cutoff) 
            end
        end
        ret = cutOffFac .* (activityCoefficients.c0 .+ cxN .* xOffs .+ cyN .* yOffs .+ cxyN .* xOffs .* yOffs .+ cxxN .* xOffs.^2 .+ cyyN .* yOffs.^2)
    
        for o = 1:length(offSets)
            xOffsN = xOffs .+ offSets[o][1]
            yOffsN = yOffs .+ offSets[o][2]
            rMagN = sqrt.(xOffsN.^2 .+ yOffsN.^2)
            cutOffFac .= 0.0
            @inline for j in 1:grid.Ny, i in 1:grid.Nx
                if rMagN[i,j] < thresh
                    cutOffFac[i,j] = TanhBump(rMagN[i,j], width, cutoff)
                end
            end
            ret .+= cutOffFac .* (activityCoefficients.c0 .+ cxN .* xOffsN .+ cyN .* yOffsN .+ cxyN .* xOffsN .* yOffsN .+ cxxN .* xOffsN.^2 .+ cyyN .* yOffsN.^2)
    
        end
    
        return ret
    end

    function ActivityFromCoefficientsSin(grid, offSets, xOffs, yOffs, activityCoefficients, tFac = 5)
        eps = 1e-6
        rMag = sqrt.(xOffs.^2 .+ yOffs.^2)
        cutOffFac = zeros(size(rMag))
        thresh = activityCoefficients.cutoff + tFac * activityCoefficients.width
        width = activityCoefficients.width
        cutoff = activityCoefficients.cutoff
        @inline for j in 1:grid.Ny, i in 1:grid.Nx
            if rMag[i,j] < thresh
                cutOffFac[i,j] = TanhBump(rMag[i,j], width, cutoff) 
            end
        end
        xOffs .+= eps 
        yOffs .+= eps
        thetas = [atan(yOffs[i,j], xOffs[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny] .- activityCoefficients.theta
        ret = cutOffFac .* activityCoefficients.c0 .* 0.5 .* (cos.(activityCoefficients.m .* thetas) .+ 1)
    
        for o = 1:length(offSets)
            xOffsN = xOffs .+ offSets[o][1] .+ eps
            yOffsN = yOffs .+ offSets[o][2] .+ eps
            rMagN = sqrt.(xOffsN.^2 .+ yOffsN.^2)
            cutOffFac .= 0.0
            @inline for j in 1:grid.Ny, i in 1:grid.Nx
                if rMagN[i,j] < thresh
                    cutOffFac[i,j] = TanhBump(rMagN[i,j], width, cutoff)
                end
            end
            thetas = [atan(yOffsN[i,j], xOffsN[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny] .- activityCoefficients.theta
            ret .+= cutOffFac .* activityCoefficients.c0 .* 0.5 .* (cos.(activityCoefficients.m .* thetas) .+ 1)
        end

        return ret
    end
    
    function SetActivityFieldFromAgents!(grid, activityField, agentHandler, bcBE_X, bcBE_Y, snapped = true)
        activityField.Values .= 0.0
    
        if (bcBE_X == "pbc") && (bcBE_Y == "pbc")
            offSets = ([grid.Nx, 0], [-grid.Nx, 0], [0, grid.Ny], [0, -grid.Ny])
        elseif bcBE_X == "pbc" 
            offSets = ([grid.Nx, 0], [-grid.Nx, 0])
        elseif bcBE_Y == "pbc" 
            offSets = ([0, grid.Ny], [0, -grid.Ny])
        else 
            offSets = ()
        end
    
        eps = 1.0e-6
        xOffs = similar(activityField.Values, Float64)
        yOffs = similar(activityField.Values, Float64)
    
        for n = 1:length(agentHandler.PlusDefects)
            if snapped 
                xOffs .= repeat(collect(1:grid.Nx) .- agentHandler.PlusDefects[n].Position[1] .+ eps, 1, grid.Ny)
                yOffs .= repeat(collect(1:grid.Nx) .- agentHandler.PlusDefects[n].Position[2] .+ eps, 1, grid.Nx)'
            else 
                xOffs .= repeat(collect(1:grid.Nx) .- agentHandler.PlusDefects[n].activityCoefficients.r0x .+ eps, 1, grid.Ny)
                yOffs .= repeat(collect(1:grid.Ny) .- agentHandler.PlusDefects[n].activityCoefficients.r0y .+ eps, 1, grid.Nx)'
            end
            if agentHandler.tweezerType == "quad"
                (cxN, cyN, cxxN, cyyN, cxyN) = BasisCoefficients(agentHandler.PlusDefects[n].activityCoefficients)
                activityField.Values .+= ActivityFromCoefficientsQuad(grid, offSets, xOffs, yOffs, agentHandler.PlusDefects[n].activityCoefficients, cxN, cyN, cxxN, cyyN, cxyN)
            elseif agentHandler.tweezerType == "sin"
                activityField.Values .+= ActivityFromCoefficientsSin(grid, offSets, xOffs, yOffs, agentHandler.PlusDefects[n].activityCoefficients)
            end
        end  
    
        for n = 1:length(agentHandler.MinusDefects)
            if snapped 
                xOffs .= repeat(collect(1:grid.Nx) .- agentHandler.MinusDefects[n].Position[1] .+ eps, 1, grid.Ny)
                yOffs .= repeat(collect(1:grid.Nx) .- agentHandler.MinusDefects[n].Position[2] .+ eps, 1, grid.Nx)'
            else 
                xOffs .= repeat(collect(1:grid.Nx) .- agentHandler.MinusDefects[n].activityCoefficients.r0x .+ eps, 1, grid.Ny)
                yOffs .= repeat(collect(1:grid.Ny) .- agentHandler.MinusDefects[n].activityCoefficients.r0y .+ eps, 1, grid.Nx)'
            end
            if agentHandler.tweezerType == "quad"
                (cxN, cyN, cxxN, cyyN, cxyN) = BasisCoefficients(agentHandler.MinusDefects[n].activityCoefficients) 
                activityField.Values .+= ActivityFromCoefficientsQuad(grid, offSets, xOffs, yOffs, agentHandler.MinusDefects[n].activityCoefficients, cxN, cyN, cxxN, cyyN, cxyN)
            elseif agentHandler.tweezerType == "sin"
                activityField.Values .+= ActivityFromCoefficientsSin(grid, offSets, xOffs, yOffs, agentHandler.MinusDefects[n].activityCoefficients)
            end
        end  
    end
    

    ########################################################################################
    #
    #                               Pulling protocol
    #
    ########################################################################################

    struct PullingProtocol
        kLegList::Vector{Any}
        rExtLegList::Vector{Any}
        function PullingProtocol()
            return new([], [])
        end
    end

    struct kLeg
        k::Real
        tA::Real
        tB::Real
        a::Real
        function kLeg(k, tA, tB, a)
            return new(k, tA, tB, a)
        end
    end

    function AddkLeg(pullingProtocol, kLeg)
        push!(pullingProtocol.kLegList, kLeg)
    end

    function evalkLeg(t, kLeg)
        return kLeg.k * Misc.SmoothBump(t, kLeg.a, kLeg.tA, kLeg.tB)
    end

    function evalk(t, pullingProtocol)
        ret = evalkLeg(t, pullingProtocol.kLegList[1])
        if length(pullingProtocol.kLegList) > 1
            for kLeg in pullingProtocol.kLegList[2:end]
                ret = ret + evalkLeg(t, kLeg)
            end
        end
        return ret
    end

    struct rExtLeg
        rA::Vector{Real}
        rB::Vector{Real}
        tA::Real
        tB::Real
        a::Real
        function rExtLeg(rA, rB, tA, tB, a)
            return new(rA, rB, tA, tB, a)
        end
    end

    function AddrExtLeg(pullingProtocol, rExtLeg)
        push!(pullingProtocol.rExtLegList, rExtLeg)
    end

    function evalrExtLeg(t, rExtLeg)
        return (rExtLeg.rA .+ (rExtLeg.rB .- rExtLeg.rA) .* ((t - rExtLeg.tA) / (rExtLeg.tB - rExtLeg.tA))) .* Misc.SmoothBump(t, rExtLeg.a, rExtLeg.tA, rExtLeg.tB)
    end

    function evalrExt(t, pullingProtocol)
        ret = evalrExtLeg(t, pullingProtocol.rExtLegList[1])
        if length(pullingProtocol.rExtLegList) > 1
            for rExtLeg in pullingProtocol.rExtLegList[2:end]
                ret .= ret .+ evalrExtLeg(t, rExtLeg)
            end
        end
        return ret
    end

    struct ExternalForceTimeSchedule
        Values::Array{Real}

        function ExternalForceTimeSchedule(timeDomain, func) # takes a generic function as second argument
            vals = map(x -> func(x), timeDomain)
            return new(vals)
        end

        function ExternalForceTimeSchedule(vals) # takes a generic function as second argument
            return new(vals)
        end

    end

    struct ExternalForce2D
        SpatialPattern
        TimeSchedule::ExternalForceTimeSchedule

        function ExternalForce2D(spatialPattern, timeSchedule::ExternalForceTimeSchedule)
            return new(spatialPattern, timeSchedule)
        end

    end


    ########################################################################################
    #
    #                               Initialization functions
    #
    ########################################################################################

    function InitializeDefectPair(grid, PInit, offX, offY, q, dec = false)

        centerX = grid.Nx / 2
        centerY = grid.Ny / 2
        nematicSoA = TensorSoA2D(grid, 0.0);
        for i=1:grid.Nx, j=1:grid.Ny

            thep = atan(j - centerY - offY - 0.5, i - centerX - offX - 0.5)
            them = atan(j - centerY + offY - 0.5, i - centerX + offX - 0.5)
            theta = 0.5 * (thep - them) + q
            if dec != false
                theta = theta * exp(-((j-centerY)^2  ) / (2*dec^2))# / (dec * sqrt(2*pi))
            end
            nVec = [cos(theta), sin(theta)]

            QTens = BerisEdwards.GetTensorFromDirector(PInit, nVec)
            nematicSoA.XXValues[i,j] = QTens[1,1] 
            nematicSoA.XYValues[i,j] = QTens[1,2] 
            nematicSoA.YXValues[i,j] = QTens[2,1] 
            nematicSoA.YYValues[i,j] = QTens[2,2] 
        end


        return nematicSoA
    end 

    function InitializeDefect(grid, PInit, offX, offY, s, q)

        centerX = grid.Nx / 2
        centerY = grid.Ny / 2
        nematicSoA = TensorSoA2D(grid);
        for i=1:grid.Nx, j=1:grid.Ny

            the = atan(j - centerY - offY - 0.5, i - centerX - offX - 0.5)
            theta = s * 0.5 * the + q
            nVec = [cos(theta), sin(theta)]

            QTens = BerisEdwards.GetTensorFromDirector(PInit, nVec)
            nematicSoA.XXValues[i,j] = QTens[1,1] 
            nematicSoA.XYValues[i,j] = QTens[1,2] 
            nematicSoA.YXValues[i,j] = QTens[2,1] 
            nematicSoA.YYValues[i,j] = QTens[2,2] 
        end


        return nematicSoA
    end 

    function InitializeWave(grid, PInit, ky, kx, off, mag)
        nematicSoA = TensorSoA2D(grid);
        for i=1:grid.Nx, j=1:grid.Ny

            theta = off + mag * (sin(i * 2 * pi * kx / (grid.Nx )) + sin(j * 2 * pi * ky / (grid.Nx )))
            nVec = [cos(theta), sin(theta)]

            QTens = BerisEdwards.GetTensorFromDirector(PInit, nVec)
            nematicSoA.XXValues[i,j] = QTens[1,1] 
            nematicSoA.XYValues[i,j] = QTens[1,2] 
            nematicSoA.YXValues[i,j] = QTens[2,1] 
            nematicSoA.YYValues[i,j] = QTens[2,2] 
        end
        return nematicSoA
    end 


    function GenerateTrajectoryX(grid, parameters, nematicSoA, v, tMax)
        nematicArray = [deepcopy(nematicSoA)]
        fldg = BerisEdwards.FreeEnergy(grid, nematicSoA, parameters["A0BE"], parameters["UBE"], parameters["LBE"], 3, parameters["bcBE_X"], parameters["bcBE_Y"]) 
        fldgArray = [deepcopy(fldg)]

        # interpolate the target functions
        xObservables = 0:1:(grid.Nx+1)
        yObservables = 0:1:(grid.Ny+1)
        obs = (xObservables, yObservables)

        XXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XXValues), Gridded(Linear()))
        XYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.XYValues), Gridded(Linear()))
        YXFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YXValues), Gridded(Linear()))
        YYFunc = interpolate(obs, ExpandForInterpolation(nematicSoA.YYValues), Gridded(Linear()))
        fldgFunc = interpolate(obs, ExpandForInterpolation(fldg.Values), Gridded(Linear()))
        
        for t in 1:(tMax - 1)
            newNematicSoA = deepcopy(nematicSoA)
            newfldg = deepcopy(fldg)
            for i=1:grid.Nx, j=1:grid.Ny
                newI = mod1(i - v * t, grid.Nx)
                newNematicSoA.XXValues[i,j] = XXFunc(newI,j)
                newNematicSoA.XYValues[i,j] = XYFunc(newI,j)
                newNematicSoA.YXValues[i,j] = YXFunc(newI,j)
                newNematicSoA.YYValues[i,j] = YYFunc(newI,j)

                newfldg.Values[i,j] = fldgFunc(newI,j)
            end
            push!(nematicArray, newNematicSoA)
            push!(fldgArray, newfldg)
        end
        return (nematicArray, fldgArray)
    end 

   

end
