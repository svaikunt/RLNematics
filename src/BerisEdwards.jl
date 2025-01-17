module BerisEdwards

    include("SharedStructs.jl")
    using .SharedStructs

    include("MathFunctions.jl")
    using .MathFunctions

    struct BEQParams
        lambda::Real
        A0::Real
        U::Real
        L::Real
        Gamma::Real
        d::Real
        function BEQParams(lambda, A0, U, L, Gamma, d = 3)
            return new(lambda, A0, U, L, Gamma, d)
        end
    end

    struct ActiveParams
        zeta::Real
        activityTimeOn::Real
        function ActiveParams(zeta, activityTimeOn = 0)
            return new(zeta, activityTimeOn)
        end
    end

    function GetTensorFromDirector(q, nVec, d = 3)

        xx = q * (nVec[1]^2 - (1/d))
        xy = q * (nVec[1]*nVec[2])
        yy = q * (nVec[2]^2 - (1/d))

        return [xx xy; xy yy]
    end

    function GetMagntiudeFromTensor(a, b, c) # a = Q_{xx}, b = Q_{xy} = Q_{yx}, c = Q_{yy}
        return 3 * (a + c)
    end

    function MSign(x, eps)
        if abs(x) < eps
            return 0.0
        else
            return sign(x)
        end
    end 

    function GetMagntiudeFromTensor2DSoA(grid, nematicSoA)

        magSoA = ScalarSoA2D(grid)
        for j = 1:grid.Ny, i = 1:grid.Nx
            magSoA.Values[i,j] = GetMagntiudeFromTensor(nematicSoA.XXValues[i,j], nematicSoA.XYValues[i,j], nematicSoA.YYValues[i,j])
        end

        return magSoA 
    end

    function GetUnitDirectorFromTensor(a, b, c) # a = Q_{xx}, b = Q_{xy} = Q_{yx}, c = Q_{yy}
        eps = 1e-8
        q = GetMagntiudeFromTensor(a, b, c)
        aq = a / q
        if  aq <= (-1/3)
            Px = 0.0
            Py = 1.0
        elseif aq >= (2/3)
            Px = 1.0
            Py = 0.0
        else 
            Px = sign(b) * sqrt((1/3) + aq)
            Py = sqrt((2/3) - aq)
        end

        return [Px, Py]
    end

    function GetDirectorFromTensor(a, b, c) # a = Q_{xx}, b = Q_{xy} = Q_{yx}, c = Q_{yy}
        eps = 1e-8
        q = GetMagntiudeFromTensor(a, b, c)
        aq = a / q
        if  aq <= (-1/3)
            Px = 0.0
            Py = 1.0
        elseif aq >= (2/3)
            Px = 1.0
            Py = 0.0
        else 
            Px = sign(b) * sqrt((1/3) + aq)
            Py = sqrt((2/3) - aq)
        end

        return q .* [Px, Py]
    end

    function GetUnitDirectorFromTensor2DSoA(grid, nematicSoA) 

        dirSoA = VectorSoA2D(grid)

        for j = 1:grid.Ny, i = 1:grid.Nx
            vec = GetUnitDirectorFromTensor(nematicSoA.XXValues[i,j], nematicSoA.XYValues[i,j], nematicSoA.YYValues[i,j])
            dirSoA.XValues[i,j] = vec[1]
            dirSoA.YValues[i,j] = vec[2]
        end

        return dirSoA 
    end 

    function GetDirectorFromTensor2DSoA(grid, nematicSoA)

        dirSoA = VectorSoA2D(grid)

        for j = 1:grid.Ny, i = 1:grid.Nx
            vec = GetDirectorFromTensor(nematicSoA.XXValues[i,j], nematicSoA.XYValues[i,j], nematicSoA.YYValues[i,j])
            dirSoA.XValues[i,j] = vec[1]
            dirSoA.YValues[i,j] = vec[2]
        end

        return dirSoA 

    end


    function GetQzzSoA(grid, nematicSoA)

        QzzSoA = GetMagntiudeFromTensor2DSoA(grid, nematicSoA)
        QzzSoA.Values .= -QzzSoA.Values ./ 3
        
        return QzzSoA 
    end


    ########################################################################################
    #
    #                    Functions and structs for Q model
    #
    ########################################################################################

    function EricksenStressTensorQ2DSoA(grid, nematicSoA, beParams, bcx, bcy)

        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]

        QPlusIdentitySoA = AddTensorSoA2D(grid, nematicSoA, IdentityMatrixSoA2D(grid, 1 / beParams.d)) # Q_{ij} + (1/d) I_{ij}
        HSoA = HTens2DSoA(grid, nematicSoA, beParams.A0, beParams.U, beParams.L, beParams.d, bcDerivX, bcDerivY) # H_{ij}

        QDotDotH = MatrixDoubleDotMatrixOnSoA2D(grid, nematicSoA, HSoA) # Q_{ij} H_{ij}
        termASoA = MultiplyTensorByScalarSoA2D(grid, QPlusIdentitySoA, QDotDotH) 
        MultiplyTensorSoA2D!(termASoA, 2 * beParams.lambda) # 2 * lambda * (Q_{ij} + (1/d) I_{ij}) * (Q_{kl}H_{kl})

        termBSoA = MatrixDotMatrixOnSoA2D(grid, HSoA, QPlusIdentitySoA)
        MultiplyTensorSoA2D!(termBSoA, - beParams.lambda) # - lambda * H_{ik} (Q_{kj} + (1/d) I_{kj})

        termCSoA = MatrixDotMatrixOnSoA2D(grid, QPlusIdentitySoA, HSoA)
        MultiplyTensorSoA2D!(termCSoA, - beParams.lambda) # - lambda * (Q_{ik} + (1/d) I_{ik}) H_{kj} 

        termDSoA = MatrixDotMatrixOnSoA2D(grid, nematicSoA, HSoA) # Q_{ik} H_{kj}

        termESoA = MatrixDotMatrixOnSoA2D(grid, HSoA, nematicSoA) 
        MultiplyTensorSoA2D!(termESoA, - 1.0) # - H_{ik} Q_{kj}

        QzzSoA = GetQzzSoA(grid, nematicSoA) # Q_{zz}

        dxQxx = bcDerivX(nematicSoA.XXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQxy = bcDerivX(nematicSoA.XYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQyx = bcDerivX(nematicSoA.YXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQyy = bcDerivX(nematicSoA.YYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQzz = bcDerivX(QzzSoA.Values, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dyQxx = bcDerivY(nematicSoA.XXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQxy = bcDerivY(nematicSoA.XYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQyx = bcDerivY(nematicSoA.YXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQyy = bcDerivY(nematicSoA.YYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQzz = bcDerivY(QzzSoA.Values, FiniteDifferenceY) ./ (2.0 * grid.dx)

        termFsSoA = ScalarSoA2D((beParams.L / 2 ) .* ((dxQxx .* dxQxx) .+ (dxQxy .* dxQxy) .+ (dxQyx .* dxQyx) .+ (dxQyy .* dxQyy) .+ (dxQzz .* dxQzz) .+
                    (dyQxx .* dyQxx) .+ (dyQxy .* dyQxy) .+ (dyQyx .* dyQyx) .+ (dyQyy .* dyQyy) .+ (dyQzz .* dyQzz)))
        termFSoA = MultiplyTensorByScalarSoA2D(grid, IdentityMatrixSoA2D(grid), termFsSoA)  # (L/2) I_{ij} * (d_k Q_{lm})^2, from pressure term

        xxGValues = (dxQxx .* dxQxx) .+ (dxQxy .* dxQxy) .+ (dxQyx .* dxQyx) .+ (dxQyy .* dxQyy) .+ (dxQzz .* dxQzz)
        xyGValues = (dxQxx .* dyQxx) .+ (dxQxy .* dyQxy) .+ (dxQyx .* dyQyx) .+ (dxQyy .* dyQyy) .+ (dxQzz .* dyQzz)
        yxGValues = (dyQxx .* dxQxx) .+ (dyQxy .* dxQxy) .+ (dyQyx .* dxQyx) .+ (dyQyy .* dxQyy) .+ (dyQzz .* dxQzz)
        yyGValues = (dyQxx .* dyQxx) .+ (dyQxy .* dyQxy) .+ (dyQyx .* dyQyx) .+ (dyQyy .* dyQyy) .+ (dyQzz .* dyQzz)
        termGSoA = TensorSoA2D(xxGValues, xyGValues, yxGValues, yyGValues)
        MultiplyTensorSoA2D!(termGSoA, - beParams.L) # - L * (d_i Q_{kl}) * (d_j Q_{kl})

        fdG = LdGEnergy(grid, nematicSoA, beParams.A0, beParams.U, beParams.L, beParams.d)
        termHSoA = MultiplyTensorByScalarSoA2D(grid, IdentityMatrixSoA2D(grid), fdG) # contribution from bulk energy

        AddTensorSoA2D!(termASoA, termBSoA) # add in place to first term
        AddTensorSoA2D!(termASoA, termCSoA)
        AddTensorSoA2D!(termASoA, termDSoA)
        AddTensorSoA2D!(termASoA, termESoA)
        AddTensorSoA2D!(termASoA, termFSoA)
        AddTensorSoA2D!(termASoA, termGSoA)
        AddTensorSoA2D!(termASoA, termHSoA)
       
        return termASoA
    end

    function ActiveStressTensorQ2DSoA(nematicSoA, grid, activityField)
        multField = ScalarSoA2D(activityField.Values .* - 1.0)
        return  MultiplyTensorByScalarSoA2D(grid, nematicSoA, multField)
    end

    function LdGEnergy(grid, nematicSoA, A0, U, L, d) # Landau de-Gennes energy

        coef1 = A0 * (1 - U / 3) / 2
        coef2 = A0 * U / 3
        coef3 = A0 * U / 4

        trSqSoA = MatrixDoubleDotMatrixOnSoA2D(grid, nematicSoA, nematicSoA) # Q_{ij} Q_{ij}
        QzzSoA = GetQzzSoA(grid, nematicSoA) 
        trSqSoA.Values .= trSqSoA.Values .+ (QzzSoA.Values .* QzzSoA.Values)
        termA = coef1 .* trSqSoA.Values

        Qsq = MatrixDotMatrixOnSoA2D(grid, nematicSoA, nematicSoA)
        trCubedSoA = MatrixDoubleDotMatrixOnSoA2D(grid, nematicSoA, Qsq)
        trCubedSoA.Values .= trCubedSoA.Values .+ (QzzSoA.Values .* QzzSoA.Values .* QzzSoA.Values)
        termB = - coef2 .* trCubedSoA.Values

        termC = coef3 .* (trSqSoA.Values .* trSqSoA.Values)

        return ScalarSoA2D(termA .+ termB .+ termC)
    end

    function FreeEnergy(grid, nematicSoA, A0, U, L, d, bcx, bcy)
        
        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]

        QzzSoA = GetQzzSoA(grid, nematicSoA)

        dxQxx = bcDerivX(nematicSoA.XXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQxy = bcDerivX(nematicSoA.XYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQyx = bcDerivX(nematicSoA.YXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQyy = bcDerivX(nematicSoA.YYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dxQzz = bcDerivX(QzzSoA.Values, FiniteDifferenceX) ./ (2.0 * grid.dx)
        dyQxx = bcDerivY(nematicSoA.XXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQxy = bcDerivY(nematicSoA.XYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQyx = bcDerivY(nematicSoA.YXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQyy = bcDerivY(nematicSoA.YYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)
        dyQzz = bcDerivY(QzzSoA.Values, FiniteDifferenceY) ./ (2.0 * grid.dx)

        bendingTerm = ScalarSoA2D((L / 2 ) .* ((dxQxx .* dxQxx) .+ (dxQxy .* dxQxy) .+ (dxQyx .* dxQyx) .+ (dxQyy .* dxQyy) .+ (dxQzz .* dxQzz) .+
        (dyQxx .* dyQxx) .+ (dyQxy .* dyQxy) .+ (dyQyx .* dyQyx) .+ (dyQyy .* dyQyy) .+ (dyQzz .* dyQzz)))
        
        ldg = LdGEnergy(grid, nematicSoA, A0, U, L, d)

        return ScalarSoA2D(0 .* bendingTerm.Values + ldg.Values)
    end

    function HTens2DSoA(grid, nematicSoA, A0, U, L, d, bcDerivX, bcDerivY) # returns - d F / d Q_{ij} + (1/3) I_{ij} d F / d Q_{kk}

        coef1 = A0 * (1 - U / 3)
        coef2 = A0 * U 
        coef3 = L

        termASoA = MultiplyTensorSoA2D(grid, nematicSoA, -coef1) # A0 * (1 - U/3) * Q_{ij}

        trSqSoA =  MatrixDoubleDotMatrixOnSoA2D(grid, nematicSoA, nematicSoA) # Q_{ij} Q_{ij}
        QzzSoA = GetQzzSoA(grid, nematicSoA) 
        trSqSoA.Values .= trSqSoA.Values .+ (QzzSoA.Values .* QzzSoA.Values)
        IQsq = MultiplyTensorByScalarSoA2D(grid, IdentityMatrixSoA2D(grid, 1/d), trSqSoA) # (I/d) * Q_{ij} Q_{ij}
        Qsq = MatrixDotMatrixOnSoA2D(grid, nematicSoA, nematicSoA) # Q_{ik} Q_{kj}
        termBSoA = SubtractTensorSoA2D(grid, Qsq, IQsq) # Q^2 - (I/d) * Q_{ij} Q_{ij}
        MultiplyTensorSoA2D!(termBSoA, coef2) # - A0 * U  * termBSoA

        termCSoA = MultiplyTensorByScalarSoA2D(grid, nematicSoA, trSqSoA)
        MultiplyTensorSoA2D!(termCSoA, -coef2) # A0 * U * tr(Q^2) * Q_{ij}

        xxDValues = coef3 .* (bcDerivX(nematicSoA.XXValues, FiniteSecondDifferenceX) .+ bcDerivY(nematicSoA.XXValues, FiniteSecondDifferenceY)) ./ ((grid.dx)^2)
        xyDValues = coef3 .* (bcDerivX(nematicSoA.XYValues, FiniteSecondDifferenceX) .+ bcDerivY(nematicSoA.XYValues, FiniteSecondDifferenceY)) ./ ((grid.dx)^2)
        yxDValues = coef3 .* (bcDerivX(nematicSoA.YXValues, FiniteSecondDifferenceX) .+ bcDerivY(nematicSoA.YXValues, FiniteSecondDifferenceY)) ./ ((grid.dx)^2)
        yyDValues = coef3 .* (bcDerivX(nematicSoA.YYValues, FiniteSecondDifferenceX) .+ bcDerivY(nematicSoA.YYValues, FiniteSecondDifferenceY)) ./ ((grid.dx)^2)
        termDSoA = TensorSoA2D(xxDValues, xyDValues, yxDValues, yyDValues) # comes from (d_k Q_{ij})^2 term

        AddTensorSoA2D!(termASoA, termBSoA) # add in place to first term
        AddTensorSoA2D!(termASoA, termCSoA)
        AddTensorSoA2D!(termASoA, termDSoA) 
     
        return termASoA
    end

    function STens2DSoA(grid, velocitySoA, nematicSoA, OmegaSoA, PsiSoA, lambda, d)

        QPlusIdentitySoA = AddTensorSoA2D(grid, nematicSoA, IdentityMatrixSoA2D(grid, 1/d)) # Q_{ij} + (1/d) I_{ij}
        lambdaPsiSoA = MultiplyTensorSoA2D(grid, PsiSoA, lambda) # lambda * Psi_{ij}
        lambdaPsiMinusOmegaSoA = SubtractTensorSoA2D(grid, lambdaPsiSoA, OmegaSoA) # lambda * Psi_{ij} - Omega_{ij}
        lambdaPsiPlusOmegaSoA = AddTensorSoA2D(grid, lambdaPsiSoA, OmegaSoA) # lambda * Psi_{ij} + Omega_{ij}
        termASoA = MatrixDotMatrixOnSoA2D(grid, lambdaPsiMinusOmegaSoA, QPlusIdentitySoA)
        termBSoA = MatrixDotMatrixOnSoA2D(grid, QPlusIdentitySoA, lambdaPsiPlusOmegaSoA)

        gradVSoA = AddTensorSoA2D(grid, OmegaSoA, PsiSoA) # d_i v_j
        QDotDotGradVSoA = MatrixDoubleDotMatrixOnSoA2D(grid, nematicSoA, gradVSoA) # Q_{ij} d_i v_j
        QDotDotGradVSoA.Values .= - (2 * lambda) .* QDotDotGradVSoA.Values 
        termCSoA =  MultiplyTensorByScalarSoA2D(grid, QPlusIdentitySoA, QDotDotGradVSoA)

        AddTensorSoA2D!(termASoA, termBSoA) # add in place to first term
        AddTensorSoA2D!(termASoA, termCSoA)

        return termASoA
    end

    function AdvectionTermSoA(grid, velocitySoA, nematicSoA, bcDerivX, bcDerivY) # returns v_k d_k Q_{ij}

        vxdxQxx = velocitySoA.XValues .* bcDerivX(nematicSoA.XXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        vydyQxx = velocitySoA.YValues .* bcDerivY(nematicSoA.XXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)

        vxdxQxy = velocitySoA.XValues .* bcDerivX(nematicSoA.XYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        vydyQxy = velocitySoA.YValues .* bcDerivY(nematicSoA.XYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)

        vxdxQyx = velocitySoA.XValues .* bcDerivX(nematicSoA.YXValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        vydyQyx = velocitySoA.YValues .* bcDerivY(nematicSoA.YXValues, FiniteDifferenceY) ./ (2.0 * grid.dx)

        vxdxQyy = velocitySoA.XValues .* bcDerivX(nematicSoA.YYValues, FiniteDifferenceX) ./ (2.0 * grid.dx)
        vydyQyy = velocitySoA.YValues .* bcDerivY(nematicSoA.YYValues, FiniteDifferenceY) ./ (2.0 * grid.dx)

        xxValues = (vxdxQxx .+ vydyQxx)
        xyValues = (vxdxQxy .+ vydyQxy)
        yxValues = (vxdxQyx .+ vydyQyx)
        yyValues = (vxdxQyy .+ vydyQyy)
 
        return TensorSoA2D(xxValues, xyValues, yxValues, yyValues)
    end

    function ComputeRHS(grid, velocitySoA, nematicSoA, OmegaSoA, PsiSoA, beParams, bcDerivX, bcDerivY) # returns S_{ij} - Gamma * H_{ij} - v_k d_k Q_{ij}

        rhs = STens2DSoA(grid, velocitySoA, nematicSoA, OmegaSoA, PsiSoA, beParams.lambda, beParams.d)
        HTensSoA = HTens2DSoA(grid, nematicSoA, beParams.A0, beParams.U, beParams.L, beParams.d, bcDerivX, bcDerivY)
        MultiplyTensorSoA2D!(HTensSoA, beParams.Gamma)
        adTermSoA = AdvectionTermSoA(grid, velocitySoA, nematicSoA, bcDerivX, bcDerivY)
        AddTensorSoA2D!(rhs, HTensSoA)
        SubtractTensorSoA2D!(rhs, adTermSoA) 

        return rhs
    end

    function PredictorCorrectorStepQ2DSoA!(grid, OmegaSoA, PsiSoA, velocitySoA, nematicSoA, dt, beParams, bcx, bcy)

        bcDerivX = BCDerivDict[bcx]
        bcDerivY = BCDerivDict[bcy]
        b = GetBoundariesFromConditions(grid, bcx, bcy)

        predNematicSoA = deepcopy(nematicSoA)
        
        rhs1 =  ComputeRHS(grid, velocitySoA, predNematicSoA, OmegaSoA, PsiSoA, beParams, bcDerivX, bcDerivY)
        predNematicSoA.XXValues[b[1]:b[2],b[3]:b[4]] .= predNematicSoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1.XXValues[b[1]:b[2],b[3]:b[4]]
        predNematicSoA.XYValues[b[1]:b[2],b[3]:b[4]] .= predNematicSoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1.XYValues[b[1]:b[2],b[3]:b[4]]
        predNematicSoA.YXValues[b[1]:b[2],b[3]:b[4]] .= predNematicSoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1.YXValues[b[1]:b[2],b[3]:b[4]]
        predNematicSoA.YYValues[b[1]:b[2],b[3]:b[4]] .= predNematicSoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ dt .* rhs1.YYValues[b[1]:b[2],b[3]:b[4]]

        rhs2 =  ComputeRHS(grid, velocitySoA, predNematicSoA, OmegaSoA, PsiSoA, beParams, bcDerivX, bcDerivY)
        nematicSoA.XXValues[b[1]:b[2],b[3]:b[4]] .= nematicSoA.XXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1.XXValues[b[1]:b[2],b[3]:b[4]] .+ rhs2.XXValues[b[1]:b[2],b[3]:b[4]]))
        nematicSoA.XYValues[b[1]:b[2],b[3]:b[4]] .= nematicSoA.XYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1.XYValues[b[1]:b[2],b[3]:b[4]] .+ rhs2.XYValues[b[1]:b[2],b[3]:b[4]]))
        nematicSoA.YXValues[b[1]:b[2],b[3]:b[4]] .= nematicSoA.YXValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1.YXValues[b[1]:b[2],b[3]:b[4]] .+ rhs2.YXValues[b[1]:b[2],b[3]:b[4]]))
        nematicSoA.YYValues[b[1]:b[2],b[3]:b[4]] .= nematicSoA.YYValues[b[1]:b[2],b[3]:b[4]] .+ (0.5 .* dt .* (rhs1.YYValues[b[1]:b[2],b[3]:b[4]] .+ rhs2.YYValues[b[1]:b[2],b[3]:b[4]]))
    end

end
