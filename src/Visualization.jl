module Visualization

    include("MathFunctions.jl")
    using .MathFunctions

    include("SharedStructs.jl")
    using .SharedStructs

    include("Analysis.jl")
    include("Training.jl")

    include("BerisEdwards.jl")

    using Interpolations
    using GLMakie
    using FFTW
    using Statistics
    using ColorTypes



    ########################################################################################
    #
    #                               Helper visualization functions
    #
    ########################################################################################
    
    FVF(vec) = Vector{Float64}(vec)
    # function FVF(vec)
    #     retVec = []
    #     #println(vec)
    #     for n in 1:length(vec[1:10])
    #         push!(retVec, Float64(vec[n]))
    #     end
    #     return retVec
    # end
    # #eturn [Float{64}(vec[n]]


    function GetDens(t, grid, col, nematicArray, activityArray, velocityArray, parameters)
        if col == "dens" # density
            dens = Observable(densityArray[t].Values)
        elseif col == "nemO" # Q magnitude
            dens = Observable((BerisEdwards.GetMagntiudeFromTensor2DSoA(grid, nematicArray[t])).Values)
        elseif col == "nemD" # Q orientation
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            dens = Observable(Analysis.thetaFromNemSoA(vecSoA.XValues, vecSoA.YValues))
        elseif col == "vor" # vorticity
            dens = Observable(Analysis.VorticitySoA(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]))
        elseif col == "speed" # velocity magnitude
            dens = Observable(sqrt.(DotProductOnSoA2D(grid, velocityArray[t], velocityArray[t]).Values))
        elseif col == "diss" # viscous dissipation
            dens = Observable(Analysis.ViscousDissipation(grid,  velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]).Values)
        elseif col == "divV" # velocity divergence
            dens = Observable(Analysis.DivV(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]))
        elseif col == "act" # velocity divergence
            dens = Observable(activityArray[t].Values)
        else
            error("Color argument not recognized.")
        end
        return dens
    end

    function UpdateDens!(t, grid, dens, col, nematicArray, activityArray, velocityArray, parameters)
        if col == "dens"
            dens[] = densityArray[t].Values
        elseif col == "nemO"
            dens[] = (BerisEdwards.GetMagntiudeFromTensor2DSoA(grid, nematicArray[t])).Values
        elseif col == "nemD"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            dens[] = Analysis.thetaFromNemSoA(vecSoA.XValues, vecSoA.YValues)
        elseif col == "vor"
            dens[] = Analysis.VorticitySoA(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3]) 
        elseif col == "divV"
            dens[] = Analysis.DivV(grid, velocityArray[t], parameters["bcLB_mX"][1:3], parameters["bcLB_mY"][1:3])
        elseif col == "act" # velocity divergence
            dens[] = activityArray[t].Values
        end
    end


    function GetduVecs(t, grid, arrows, arrowFac, nematicArray, velocityArray, parameters)
        if arrows == "vel"
            duVecs = Observable(arrowFac * [Point2(velocityArray[t].XValues[i,j], velocityArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "nem"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            us = []
            vs = []
            for i = 1:grid.Nx
                for j = 1:grid.Ny
                    push!(us, arrowFac * vecSoA.XValues[i,j])
                    push!(vs, arrowFac * vecSoA.YValues[i,j])
                end 
            end 
            #duVecs = [Observable(us), Observable(vs)]
            duVecs = [us, vs]
        elseif arrows == "diva"
            if parameters["BEModel"] == "P"
                aSoA = BerisEdwards.ActiveStressTensorP2DSoA(directorArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            else 
                aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(nematicArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            end
            diva = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, aSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs = Observable(arrowFac * [Point2(diva.XValues[i,j], diva.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        elseif arrows == "divE"
            if parameters["BEModel"] == "P"
                beParams = BerisEdwards.BEPParams(parameters["xiBE"], parameters["alphaBE"], parameters["betaBE"], parameters["kappaBE"], parameters["GammaBE"], parameters["tauBE"])
                eSoA = BerisEdwards.EricksenStressTensorP2DSoA(grid, directorArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            else 
                beParams = BerisEdwards.BEQParams(parameters["lambdaBE"], parameters["A0BE"], parameters["UBE"], parameters["LBE"], parameters["GammaBE"])
                eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(grid, nematicArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            end
            dive = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, eSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs = Observable(arrowFac * [Point2(dive.XValues[i,j], dive.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny])
        else
            error("Arrow argument not recognized.")
        end
        return duVecs
    end

    function UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, velocityArray, nematicArray, parameters)
        if arrows == "vel"
            duVecs[] = [arrowFac * Point2(velocityArray[t].XValues[i,j], velocityArray[t].YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "nem"
            vecSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t])
            us = []
            vs = []
            for i = 1:grid.Nx
                for j = 1:grid.Ny
                    push!(us, arrowFac * vecSoA.XValues[i,j])
                    push!(vs, arrowFac * vecSoA.YValues[i,j])
                end 
            end 
            duVecs[1].val = us
            duVecs[2].val = vs
        elseif arrows == "divS"
            divS = Analysis.DivS(grid, sigmaVEArray[t], parameters["bcVE_X"], parameters["bcVE_Y"])
            duVecs[] = arrowFac * [Point2(divS.XValues[i,j], divS.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "diva"
            if parameters["BEModel"] == "P"
                aSoA = BerisEdwards.ActiveStressTensorP2DSoA(directorArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            else 
                aSoA = BerisEdwards.ActiveStressTensorQ2DSoA(nematicArray[t], grid, BerisEdwards.ActiveParams(parameters["zeta"]))
            end
            diva = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, aSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs[] = arrowFac * [Point2(diva.XValues[i,j], diva.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        elseif arrows == "divE"
            if parameters["BEModel"] == "P"
                beParams = BerisEdwards.BEPParams(parameters["xiBE"], parameters["alphaBE"], parameters["betaBE"], parameters["kappaBE"], parameters["GammaBE"], parameters["tauBE"])
                eSoA = BerisEdwards.EricksenStressTensorP2DSoA(grid, directorArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            else 
                beParams = BerisEdwards.BEQParams(parameters["lambdaBE"], parameters["A0BE"], parameters["UBE"], parameters["LBE"], parameters["GammaBE"])
                eSoA = BerisEdwards.EricksenStressTensorQ2DSoA(grid, nematicArray[t], beParams, parameters["bcBE_X"], parameters["bcBE_Y"])
            end
            dive = Analysis.DivS(grid, MatrixTransposeOnSoA2D(grid, eSoA), parameters["bcBE_X"], parameters["bcBE_Y"])
            duVecs[] = arrowFac * [Point2(dive.XValues[i,j], dive.YValues[i,j]) for i = 1:grid.Nx, j = 1:grid.Ny]
        end
    end

    function GetPts(grid)
        xs = map(x-> SharedStructs.GetGridPoint2D(x,1,grid)[1], 1:grid.Nx)
        ys = map(y-> SharedStructs.GetGridPoint2D(1,y,grid)[2], 1:grid.Ny)
        return (xs, ys, vec(Point2.(xs, ys')))
    end

    # function DecimateArrows!(grid, pts, duVecs, skip, halfOff = false)
    #     if (!halfOff)
    #         nxs = map(x-> SharedStructs.GetGridPoint2D(x,1,grid)[1], 1:skip:grid.Nx)
    #         nys = map(y-> SharedStructs.GetGridPoint2D(1,y,grid)[2], 1:skip:grid.Ny)
    #         duVecs[] = duVecs[][1:skip:end, 1:skip:end]
    #         return vec(Point2.(nxs, nys'))
    #     else 
    #         nxs = []
    #         nys = []
    #         nus = []
    #         nvs = []
    #         for i = 1:skip:grid.Nx
    #             for j = 1:skip:grid.Ny
    #                 push!(nxs, SharedStructs.GetGridPoint2D(i,j,grid)[1] - 0.5 * duVecs[1][][(i-1) * grid.Ny + j])
    #                 push!(nys, SharedStructs.GetGridPoint2D(i,j,grid)[2] - 0.5 * duVecs[2][][(i-1) * grid.Ny + j])
    #                 push!(nus, duVecs[1][][(i-1) * grid.Ny + j])
    #                 push!(nvs, duVecs[2][][(i-1) * grid.Ny + j])
    #             end 
    #         end 
    #         duVecs[1].val = nus
    #         duVecs[2].val = nvs
    #         return [nxs, nys]
    #     end 
    # end

    function DecimateArrows!(grid, pts, duVecs, skip, halfOff = false)
        if (!halfOff)
            nxs = map(x-> SharedStructs.GetGridPoint2D(x,1,grid)[1], 1:skip:grid.Nx)
            nys = map(y-> SharedStructs.GetGridPoint2D(1,y,grid)[2], 1:skip:grid.Ny)
            duVecs = duVecs[1:skip:end, 1:skip:end]
            return vec(Point2.(nxs, nys'))
        else 
            nxs = []
            nys = []
            nus = []
            nvs = []
            for i = 1:skip:grid.Nx
                for j = 1:skip:grid.Ny
                    push!(nxs, SharedStructs.GetGridPoint2D(i,j,grid)[1] - 0.5 * duVecs[1][(i-1) * grid.Ny + j])
                    push!(nys, SharedStructs.GetGridPoint2D(i,j,grid)[2] - 0.5 * duVecs[2][(i-1) * grid.Ny + j])
                    push!(nus, duVecs[1][(i-1) * grid.Ny + j])
                    push!(nvs, duVecs[2][(i-1) * grid.Ny + j])
                end 
            end 
            duVecs[1] = nus
            duVecs[2] = nvs
            return [nxs, nys]
        end 
    end

    function DrawHeatMap!(scene, xs, ys, dens, col, colorrange)
        if col == "pol"
            #GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (-pi/2, pi/2), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (-pi, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        elseif col == "nemD"
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange = (0, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        else
            GLMakie.heatmap!(scene, xs, ys, dens, colorrange=colorrange, interpolate = true) # (-5e-4, 5e-4)
        end
    end

    function DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        if col == "pol"
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (pi/4 - 0.5 , pi/4 + 0.5), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
            #hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (-pi, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        elseif col == "nemD"
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange = (0, pi), colormap = range(LCHab(70,70,0), stop=LCHab(70,70,360), length=90))
        else
            hm = GLMakie.heatmap!(xs, ys, dens, colorrange=colorrange, interpolate = true) # (-5e-4, 5e-4)
        end
        return hm
    end

    function SetLims!(scene, grid)
        #Makie.xlims!(scene, 0, grid.Nx+1)
        #Makie.ylims!(scene, 0, grid.Ny+1)
        Makie.xlims!(0, grid.Nx+1)
        Makie.ylims!(0, grid.Ny+1)
    end

    function SetLimsNS!(ax, grid, off = 0)
        Makie.xlims!(ax, 0 + off, grid.Nx+1 - off)
        Makie.ylims!(ax, 0 + off, grid.Ny+1 - off)
    end

    GetSeparation(agentHandler) = agentHandler.PlusDefects[1].Position[1] - agentHandler.MinusDefects[1].Position[1]

    function DrawTextBox!(str1, str2; textPosition = [5, 76], textDist = 2, boff = 1.2, loff = 0.1, w = 20, h = 3, sw = 4, ts = 40)
        poly!(Rect(textPosition[1] - boff, textPosition[2] - loff, w, h), color = :white,
            strokecolor = :black, strokewidth = sw)
        GLMakie.text!(str1,
            position = (textPosition[1], textPosition[2]), fontsize = ts, font = "Arial")

        GLMakie.text!(str2,
            position = (textPosition[1], textPosition[2] - textDist), fontsize = ts, font = "Arial")
    end

    function GetStreamFunction(grid, velocityArray)
        xObservables = 1:1:grid.Nx
        yObservables = 1:1:grid.Ny
        Observables = (xObservables, yObservables)
        interpolationXFuncList = [interpolate(Observables, velocityArray[t].XValues, Gridded(Linear())) for t in 1:length(velocityArray)]
        interpolationYFuncList = [interpolate(Observables, velocityArray[t].YValues, Gridded(Linear())) for t in 1:length(velocityArray)]
        streamFunction(x::Point2, t) = Point2(
            interpolationXFuncList[t](x[1], x[2]),
            interpolationYFuncList[t](x[1], x[2]))
        return streamFunction
    end

    function GetNematicTrajectory(grid, t, nematicArray, bcx, bcy)
        posListPlusX = []
        posListPlusY = []
        posListMinX = []
        posListMinY = []
        
        for tp = 1:t
            agentHandler = Training.AgentHandler(grid, nematicArray[tp], bcx, bcy)
            push!(posListPlusX, agentHandler.PlusDefects[1].Position[1])
            push!(posListPlusY, agentHandler.PlusDefects[1].Position[2])
            push!(posListMinX, agentHandler.MinusDefects[1].Position[1])
            push!(posListMinY, agentHandler.MinusDefects[1].Position[2])
        end 

        return (Observable(Vector{Float64}(posListPlusX)), Observable(Vector{Float64}(posListPlusY)), 
            Observable(Vector{Float64}(posListMinX)), Observable(Vector{Float64}(posListMinY)))

    end 


    function GetNematicOrientation(grid, t, nematicArray, bcx, bcy)

        agentHandler = Training.AgentHandler(grid, nematicArray[t], bcx, bcy)

        orPlus = agentHandler.PlusDefects[1].Orientation
        orMint = agentHandler.MinusDefects[1].Orientation

        orMin = orMint

        fac = 4

        nemDict = Dict(

        "posPlus" => Observable([Point2(agentHandler.PlusDefects[1].Position...)]),
        "plusU" => Observable([Point2(fac * cos(orPlus), fac * sin(orPlus))]),

        "posMin" => Observable([Point2(agentHandler.MinusDefects[1].Position...)]),
        "minU1" => Observable([Point2(fac * cos(orMin), fac * sin(orMin))]),
        "minU2" => Observable([Point2(fac * cos(orMin + 2*pi/3), fac * sin(orMin + 2*pi/3))]),
        "minU3" => Observable([Point2(fac * cos(orMin + 4*pi/3), fac * sin(orMin + 4*pi/3))]),

        )

        return nemDict
    end

    function UpdateNematicOrientation!(grid, nemDict, t, nematicArray, bcx, bcy)

        agentHandler = Training.AgentHandler(grid, nematicArray[t], bcx, bcy)

        try
            orPlus = agentHandler.PlusDefects[1].Orientation
            orMin = agentHandler.MinusDefects[1].Orientation

            fac = 4

            nemDict["posPlus"][] = [Point2(agentHandler.PlusDefects[1].Position...)]
            nemDict["plusU"][] =[Point2(fac * cos(orPlus), fac * sin(orPlus))]

            nemDict["posMin"][] = [Point2(agentHandler.MinusDefects[1].Position...)]
            nemDict["minU1"][] = [Point2(fac * cos(orMin), fac * sin(orMin))]
            nemDict["minU2"][] = [Point2(fac * cos(orMin + 2*pi/3), fac * sin(orMin + 2*pi/3))]
            nemDict["minU3"][] = [Point2(fac * cos(orMin + 4*pi/3), fac * sin(orMin + 4*pi/3))]
        catch
        end

    end
        

    ########################################################################################
    #
    #                               Main visualization functions
    #
    ########################################################################################


    function AnimateTrajectoryArrows(parameters,  velocityArray, nematicArray, activityArray, agentHandlerArray;
        episodeStride = 10, tracker = false,
        arrows = "vel", col = "dens", defects = "none", arrowFac = 1e0, arrowSkip = 1, arrowWidth = 1, arrowHead = 1, arrowOp = 1,
        markersize = 20,
        colorrange = (0.99, 1.001), recording = false, moviePath = "", sleepLength = 0.1, res = 2000, ar = 1, cScaleFac = 1,
        fontsize = 35, ticks = [1], tickLabels = [""], xlabel = "", ylabel = "", clabel = "")

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        f = Figure(resolution = (res, ar * res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        dens = GetDens(1, grid, col, nematicArray, activityArray, velocityArray, parameters)
        dens[] = dens[] .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)
        duVecs = GetduVecs(1, grid, arrows, arrowFac, nematicArray, velocityArray, parameters)
        if arrows == "nem"
            halfOff = true 
        else
            halfOff = false 
        end
        npts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
        if arrows != "nem"
            arrows!(npts, duVecs, linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        else 
            apts = [Observable(npts[1]), Observable(npts[2])]
            arrows!(apts[1], apts[2], duVecs[1], duVecs[2], linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        end 

        if defects != "none"
            if defects == "Q"
                dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[1]) 
                wna = Analysis.WindingNumbersNematicPBC(grid, dirSoA)
                cut = 0.4
            else
                dirSoA = directorArray[1]
                wna = Analysis.WindingNumbers(grid, dirSoA)
                cut = 0.4
            end
            (xspV, yspV, xsmV, ysmV) = Analysis.CreateScatterFromWNA(wna, 1, cut)
            xsp = Observable(xspV)
            ysp = Observable(yspV)
            xsm = Observable(xsmV)
            ysm = Observable(ysmV)
            GLMakie.scatter!(xsp, ysp, markersize = markersize, marker = :circle, color = :black)
            GLMakie.scatter!(xsm, ysm, markersize = markersize, marker = :circle, color = :white)

            orFacA = 4
            nemDict = GetNematicOrientation(grid, 1, nematicArray, parameters["bcBE_X"], parameters["bcBE_X"])
            Makie.arrows!(nemDict["posPlus"], nemDict["plusU"],
                arrowsize = 0, linewidth = orFacA * arrowWidth, color = :black)
            Makie.arrows!(nemDict["posMin"], nemDict["minU1"],
                arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)
            Makie.arrows!(nemDict["posMin"], nemDict["minU2"],
                arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)
            Makie.arrows!(nemDict["posMin"], nemDict["minU3"],
                arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)
        end

        if tracker
            baseTextSep = "Separation: "
            pSep = Observable(baseTextSep * string(GetSeparation(agentHandlerArray[1])))

            # baseTextSep = "Orientation: "
            # pSep = Observable(baseTextSep * string(round(agentHandlerArray[1].PlusDefects[1].Orientation; digits = 2)))

            baseTextEp = "Episode: "

            episodeCount = 1
            
            pEp = Observable(baseTextEp * string(episodeCount))

            DrawTextBox!(pSep, pEp; textPosition = [5, 85], textDist = -5, boff = 1.2, loff = 0.1, w = 37, h = 10, sw = 4, ts = 20)
        end

        if recording
            record(f, moviePath, 1:length(velocityArray)) do t
                UpdateDens!(t, grid, dens, col, nematicArray, activityArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, velocityArray, nematicArray, parameters)
                nApts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
                if arrows == "nem" 
                    apts[1][] = nApts[1]
                    apts[2][] = nApts[2]
                end
                if defects != "none"
                    if defects == "Q"
                        dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t]) 
                        wna = Analysis.WindingNumbersNematicPBC(grid, dirSoA)
                        cut = 0.4
                    else
                        dirSoA = directorArray[t]
                        wna = Analysis.WindingNumbers(grid, dirSoA)
                        cut = 0.4
                    end
                    (xspV, yspV, xsmV, ysmV) = Analysis.CreateScatterFromWNA(wna, 1, cut)
                    xsp.val = xspV
                    ysp[] = yspV 
                    xsm.val = xsmV 
                    ysm[] = ysmV
                    UpdateNematicOrientation!(grid, nemDict, t, nematicArray, parameters["bcBE_X"], parameters["bcBE_X"])
   
                end

                if tracker
                    if (t != 1) && ((t - 1) % episodeStride == 0)
                        episodeCount += 1
                    end 
                    try
                        pSep[] = baseTextSep * string(GetSeparation(agentHandlerArray[t]))
                    #pSep[] = baseTextSep * string(string(round(agentHandlerArray[t].PlusDefects[1].Orientation; digits = 2)))

                        pEp[] = baseTextEp * string(episodeCount)
                    catch 
                    end
                    
                    
                end

                sleep(sleepLength)
            end
        else
            for t in 1:length(velocityArray)
                UpdateDens!(t, grid, dens, col, nematicArray, activityArray, velocityArray, parameters)
                dens[] = dens[] .* cScaleFac
                UpdateDuVecs!(t, grid, duVecs, arrows, arrowFac, nematicArray, velocityArray, parameters)
                nApts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
                if arrows == "nem"
                    apts[1][] = nApts[1]
                    apts[2][] = nApts[2]
                end
                if defects != "none"
                    if defects == "Q"
                        dirSoA = BerisEdwards.GetUnitDirector2DSoA(grid, nematicArray[t]) 
                        wna = Analysis.WindingNumbersNematicPBC(grid, dirSoA)
                        cut = 0.9
                    else
                        dirSoA = directorArray[t]
                        wna = Analysis.WindingNumbers(grid, dirSoA)
                        cut = 0.4
                    end
                    (xspV, yspV, xsmV, ysmV) =  Analysis.CreateScatterFromWNA(wna, 1, cut)
                    xsp.val = xspV
                    ysp[] = yspV 
                    xsm.val = xsmV 
                    ysm[] = ysmV
                    UpdateNematicOrientation!(grid, nemDict, t, nematicArray, parameters["bcBE_X"], parameters["bcBE_X"])
                end
                sleep(sleepLength)
                current_figure()
            end
        end
    end # AnimateTrajectoryArrows()

    function StaticArrows(t, parameters,  velocityArray, nematicArray, activityArray, agentHandlerArray;
        tracker = false,
        arrows = "vel", col = "dens", defects = "none", 
        arrowFac = 1e0, arrowSkip = 1, arrowWidth = 1, arrowHead = 1, arrowOp = 1, markersize = 20,
        colorrange = (0.99, 1.001), recording = false, imagePath = "", res = 2000, 
        off = 0, ar = 1, cScaleFac = 1, cwidth, 
        fontsize = 35, ticks = [0], tickLabels = [""], fac = 10^6, xlabel = "", ylabel = "", clabel = "",
        display = true)

        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        f = Figure(resolution = (res, ar * res), fontsize = fontsize, font = "Arial", dpi = 300)
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        if length(off) == 1
            SetLimsNS!(ax, grid, off)
        else
            SetLimsNS!(ax, grid, 0)
            Makie.xlims!(ax, off[1], off[2])
            Makie.ylims!(ax, off[3], off[4])
        end

        if defects == "Q"
            dirSoA = BerisEdwards.GetUnitDirectorFromTensor2DSoA(grid, nematicArray[t]) 
            wna = Analysis.WindingNumbersNematicPBC(grid, dirSoA)
            cut = 0.4
        else
            dirSoA = directorArray[t]
            wna = Analysis.WindingNumbers(grid, dirSoA)
            cut = 0.4
        end
        (xsp, ysp, xsm, ysm) =  Analysis.CreateScatterFromWNA(wna, 1, cut)

        dens = GetDens(t, grid, col, nematicArray, activityArray, velocityArray, parameters)
        
        dens[] = dens[] .* cScaleFac
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
     
        GLMakie.Colorbar(f[1,2], hm, label = clabel, width = cwidth)
        duVecs = GetduVecs(t, grid, arrows, arrowFac, nematicArray, velocityArray, parameters)
        if arrows == "nem"
            halfOff = true 
        else
            halfOff = false 
        end
        apts = DecimateArrows!(grid, pts, duVecs, arrowSkip, halfOff)
        if arrows != "nem"
            arrows!(apts, duVecs, linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp), aspect = 1)
        else 
            arrows!(FVF(apts[1]), FVF(apts[2]), FVF(duVecs[1]), FVF(duVecs[2]), linewidth = arrowWidth, arrowsize = arrowHead, color = (:black, arrowOp))#, aspect = 1)
        end 


        GLMakie.scatter!(xsp, ysp, markersize = markersize, marker = :square, color = :black)
        GLMakie.scatter!(xsm, ysm, markersize = markersize, marker = :square, color = :white)

        orFacA = 4
        nemDict = GetNematicOrientation(grid, t, nematicArray, parameters["bcBE_X"], parameters["bcBE_X"])
        Makie.arrows!(nemDict["posPlus"], nemDict["plusU"],
            arrowsize = 0, linewidth = orFacA * arrowWidth, color = :black)
        Makie.arrows!(nemDict["posMin"], nemDict["minU1"],
            arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)
        Makie.arrows!(nemDict["posMin"], nemDict["minU2"],
            arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)
        Makie.arrows!(nemDict["posMin"], nemDict["minU3"],
            arrowsize = 0, linewidth = orFacA * arrowWidth, color = :white)


        if tracker
            baseTextSep = "Separation: "
            pSep = Observable(baseTextSep * string(GetSeparation(agentHandlerArray[t])))

            #baseTextSep = "rₛₑₚ: "
            #baseTextSep = rich("r", subscript("sep"))
            #pSep = Observable(baseTextSep)# * string(GetSeparation(agentHandlerArray[t])))
            #pSep = baseTextSep# * string(GetSeparation(agentHandlerArray[t])))

            #baseTextSep = "Orientation: "
            #baseTextSep = "sin(ϕ₊): "
            #pSep = Observable(baseTextSep * string(round(sin(agentHandlerArray[t].PlusDefects[1].Orientation); digits = 2)))
        

            # baseTextEp = "Episode: "

            # episodeCount = 1

            baseTextEp = "Step "

            #episodeCount = 1
            
            #pEp = Observable(baseTextEp * string(t))
            pEp = baseTextEp * string(t)

            #DrawTextBox!(pSep, pEp; textPosition = [5, 85], textDist = -5, boff = 1.2, loff = 0.1, w = 37, h = 10, sw = 4, ts = 40)
            DrawTextBox!(pSep, pEp; textPosition = [5, 92-30], textDist = -5, boff = 1.2, loff = 0.1, w = 37, h = 10, sw = 3, ts = 40)
            #DrawTextBox!(pSep, pEp; textPosition = [5, 92-30], textDist = -5, boff = 1.2, loff = 0.1, w = 25, h = 10, sw = 3, ts = 40)
            #DrawTextBox!(pSep, pEp; textPosition = [5, 92-30], textDist = -5, boff = 1.2, loff = 0.1, w = 29, h = 10, sw = 3, ts = 40)
        end

        if recording
            save(imagePath, f, px_per_unit = 3)
        end

        if display
            GLMakie.current_figure()
        else
            return ax
        end

    end # StaticArrowsNS()

    function StaticAgentHandlerList(parameters,  agentHandler;
        colorrange = (0.99, 1.001), moviePath = "", sleepLength = 0.1, res = 2000, cScaleFac = 1,
        fontsize = 35, ticks = [1], tickLabels = [""], xlabel = "", ylabel = "", clabel = "", 
        display = true, recording = false)

        parameters["dx"] = 1
        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        f = Figure(resolution = (res, res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        activityField = ScalarSoA2D(grid)

        Training.SetActivityFieldFromAgents!(grid, activityField, agentHandler, parameters["bc"], parameters["bc"])

        dens = Observable(activityField.Values)
        dens[] = dens[] .* cScaleFac
        col = "act"
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)

        xspV = agentHandler.PlusDefects[1].Position[1]
        yspV = agentHandler.PlusDefects[1].Position[2]
        xsmV = agentHandler.MinusDefects[1].Position[1]
        ysmV = agentHandler.MinusDefects[1].Position[2]
        xsp = Observable(xspV)
        ysp = Observable(yspV)
        xsm = Observable(xsmV)
        ysm = Observable(ysmV)
        GLMakie.scatter!(xsp, ysp, markersize = 20, marker = :circle, color = :black)
        GLMakie.scatter!(xsm, ysm, markersize = 20, marker = :circle, color = :white)

        baseTextSep = "Separation: "
        baseTextEp = "Episode: "

        episodeCount = 1
        pSep = Observable(baseTextSep * string(GetSeparation(agentHandler)))
        pEp = Observable(baseTextEp * string(episodeCount))

        DrawTextBox!(pSep, pEp; textPosition = [5, 85], textDist = -5, boff = 1.2, loff = 0.1, w = 37, h = 10, sw = 4, ts = 40)

        if recording
            save(imagePath, f, px_per_unit = 3)
        end

        if display
            GLMakie.current_figure()
        else
            return ax
        end
    end # StaticAgentHandlerList()

    function AnimateAgentHandlerList(parameters,  agentHandlerList;
        episodeStride = 10,
        colorrange = (0.99, 1.001), moviePath = "", sleepLength = 0.1, res = 2000, cScaleFac = 1,
        fontsize = 35, ticks = [1], tickLabels = [""], xlabel = "", ylabel = "", clabel = "")

        parameters["dx"] = 1
        grid = Analysis.GetGrid(parameters)
        (xs, ys, pts) = GetPts(grid)

        f = Figure(resolution = (res, res), fontsize = fontsize, font = "Arial", dpi = 300)
        #tickLabels = [string((t - ticks[Int(ceil(length(ticks)/2))]) * parameters["Cx"] * fac ) for t in ticks]
        ax = Axis(f[1, 1],
            xlabel = xlabel,
            ylabel = ylabel,
            xticks = (ticks, tickLabels),
            yticks = (ticks, tickLabels)
        )

        activityField = ScalarSoA2D(grid)

        agentHandler = agentHandlerList[1]
        Training.SetActivityFieldFromAgents!(grid, activityField, agentHandler, parameters["bc"], parameters["bc"])

        dens = Observable(activityField.Values)
        dens[] = dens[] .* cScaleFac
        col = "act"
        hm = DrawHeatMapNS!(xs, ys, dens, col, colorrange)
        SetLimsNS!(ax, grid)
        GLMakie.Colorbar(f[1,2], hm, label = clabel)

        
        xspV = agentHandler.PlusDefects[1].Position[1]
        yspV = agentHandler.PlusDefects[1].Position[2]
        xsmV = agentHandler.MinusDefects[1].Position[1]
        ysmV = agentHandler.MinusDefects[1].Position[2]
        xsp = Observable(xspV)
        ysp = Observable(yspV)
        xsm = Observable(xsmV)
        ysm = Observable(ysmV)
        GLMakie.scatter!(xsp, ysp, markersize = 20, marker = :circle, color = :black)
        GLMakie.scatter!(xsm, ysm, markersize = 20, marker = :circle, color = :white)

        baseTextSep = "Separation: "
        baseTextEp = "Episode: "

        episodeCount = 1
        pSep = Observable(baseTextSep * string(GetSeparation(agentHandler)))
        pEp = Observable(baseTextEp * string(episodeCount))

        DrawTextBox!(pSep, pEp; textPosition = [5, 85], textDist = -5, boff = 1.2, loff = 0.1, w = 37, h = 10, sw = 4, ts = 40)


        record(f, moviePath, 1:length(agentHandlerList)) do t

            if length(agentHandlerList[t].PlusDefects) == 1
                agentHandler = agentHandlerList[t]
                Training.SetActivityFieldFromAgents!(grid, activityField, agentHandler, parameters["bc"], parameters["bc"])
                dens[] = activityField.Values
                dens[] = dens[] .* cScaleFac
                
                xspV = agentHandler.PlusDefects[1].Position[1]
                yspV = agentHandler.PlusDefects[1].Position[2]
                xsmV = agentHandler.MinusDefects[1].Position[1]
                ysmV = agentHandler.MinusDefects[1].Position[2]
                xsp.val = xspV
                ysp[] = yspV 
                xsm.val = xsmV 
                ysm[] = ysmV

                if (t != 1) && ((t - 1) % episodeStride == 0)
                    episodeCount += 1
                end 

                pSep[] = baseTextSep * string(GetSeparation(agentHandler))
                pEp[] = baseTextEp * string(episodeCount)
            end
    
            sleep(sleepLength)
        end
       
    end # AnimateTrajectoryArrows()


   

end # module
