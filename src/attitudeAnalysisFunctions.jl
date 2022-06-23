module attitudeAnalysisFunctions
using attitudeFunctions
using LinearAlgebra
using Random
using lightCurveOptimization
using lightCurveModeling
using Distributed
using NLopt
using SharedArrays
using JLD2
using MATLABfunctions
using Colors
using Infiltrator
using ProgressMeter
using Accessors

import Clustering: kmeans, kmedoids, assignments

const Vec{T<:Number} = AbstractArray{T,1}
const Mat{T<:Number} = AbstractArray{T,2}
const ArrayOfVecs{T<:Number} = Array{V,1} where V <: Vec
const ArrayofMats{T<:Number} = Array{M,1} where M <: Mat
const MatOrVecs = Union{Mat,ArrayOfVecs}
const MatOrVec = Union{Mat,Vec}
const anyAttitude = Union{Mat,Vec,DCM,MRP,GRP,quaternion}

include("utilities.jl")

export randomAttAnalysis, generateLMData, findLevelSets, monteCarloAttAnalysis, singleAttAnalysis, analyzeRandomAttitudeConvergence, optimizationComparison, multiAttAnalysis, randomStateAnalysis

function monteCarloAttAnalysis(trueState :: Vector, N :: Int64, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions(); show_progress = true)

    resultsFull = Array{LMoptimizationResults,1}(undef,N)
    if show_progress
        print("Running Optimization")
        p = Progress(N);
    end

    for i = 1:N
        resultsFull[i] = singleAttAnalysis(trueState, LMproblem, options)
        if show_progress
            sleep(.1)
            next!(p)
        end
    end

    return resultsFull
end

function _singleAttAnalysis(trueState :: Vector, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions())

    if any(options.algorithm .== [:MPSO, :MPSO_VGC, :MPSO_NVC, :PSO_cluster, :MPSO_full_state])

        results = PSO_LM(trueState, LMproblem, options)

    elseif options.algorithm == :MPSO_AVC

        costFunc = costFuncGenPSO(trueState, LMproblem, options.Parameterization, true)
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

        if options.initMethod == :random
            xinit = randomAtt(1, options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_AVC(xinit, costFunc, clusterFunc, options.optimizationParams)

    elseif any(options.algorithm .== [:LD_SLSQP, :GN_CRS2_LM, :G_MLSL, :GD_STOGO, :GN_AGS, :GN_ISRES])

        if options.initMethod == :random
            xinit = randomAtt(1, options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end

        if any(abs.(xinit) .> 1)
            xinit = sMRP(xinit)
        end
        # @infiltrate
        options = @set options.initVals = xinit
        costFunc = costFuncGenFD(trueState, LMproblem)
        (minf, minx, ret) = GB_main(costFunc, options)

        results = GB_results(minf,minx,ret)

    elseif options.algorithm == :ELD_SLSQP

        results = EGB_LM(trueState, LMproblem, options)

    else
        error("Invalid Algorithm")
    end

    return results
end

function singleAttAnalysis(trueState :: Vector, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions())

    if options.Parameterization == quaternion
        if length(trueState) == 3
            trueState = p2q(trueState, LMproblem.a, LMproblem.f)
        elseif length(trueState) == 4

        else
            error("invalid state")
        end

    elseif options.Parameterization <: Union{MRP,GRP}
        if length(trueState) == 4
            trueState = q2p(trueState, LMproblem.a, LMproblem.f)
        elseif length(trueState) == 3

        else
            error("invalid state")
        end
    end

    results = _singleAttAnalysis(trueState, LMproblem, options)

    return LMoptimizationResults(results, trueState, LMproblem, options)
end

function randomAttAnalysis(N :: Int64, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions())

    trueAttitudes = randomAtt(N, options.Parameterization, options.vectorize)

    return multiAttAnalysis(N, trueAttitudes, LMproblem, options)
end

function randomStateAnalysis(N :: Int64, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions())


    at_true = randomAtt(options.optimizationParams.N, options.Parameterization)

    if any(options.algorithm .== (:MPSO_full_state))
        w_true = randomBoundedAngularVelocity(options.optimizationParams.N, LMproblem.angularVelocityBound)

        ST_true = [[at_true[i];w_true[i]] for i in 1:length(at_true)]
    else
        ST_true = at_true
    end

    return multiAttAnalysis(N, ST_true, LMproblem, options)
end

function multiAttAnalysis(N :: Int64, states, LMproblem = LMoptimizationProblem(), options = LMoptimizationOptions())


    resultsFull = Array{LMoptimizationResults,1}(undef,N)
    print("Running Optimization")
    p = Progress(N);
    for i = 1:N

        if N>1
            x = states[i]
        elseif N == 1
            x = states
        end

        resultsFullTemp = _singleAttAnalysis(x, LMproblem, options)

        resultsFull[i] = LMoptimizationResults(resultsFullTemp, x, LMproblem, options)
        sleep(.1)
        next!(p)
    end

    return resultsFull
end

function optimizationComparison(opt1 :: LMoptimizationOptions, LMopt2 :: LMoptimizationOptions, N :: Int64, LMprob = LMoptimizationProblem())

    trueAttitudes = randomAtt(N,DCM,opt1.vectorize)
    if opt1.algorithm == :MPSO_full_state | opt2.algorithm == :MPSO_full_state
        wvec = randomBoundedAngularVelocity(N, LMprob.angularVelocityBound)
    end

    resultsFull1 = Array{LMoptimizationResults,1}(undef,N)
    resultsFull2 = Array{LMoptimizationResults,1}(undef,N)

    if opt1.Parameterization == quaternion
        f1 = (x)->A2q(x)
    elseif (opt1.Parameterization == MRP) | (opt1.Parameterization == GRP)
        f1 = (x)->A2p(x,LMprob.a,LMprob.f)
    end

    if opt2.Parameterization == quaternion
        f2 = (x)->A2q(x)
    elseif (opt2.Parameterization == MRP) | (opt2.Parameterization == GRP)
        f2 = (x)->A2p(x,LMprob.a,LMprob.f)
    end

    for i = 1:N

        if N>1
            A = trueAttitudes[i]
            w = wvec[i]
        elseif N == 1
            A = trueAttitudes
            w = wvec
        end

        if opt1.algorithm == :MPSO_full_state
            x1 = [f1(A);w]
        else
            x1 = f1(A)
        end

        rF1 = _singleAttAnalysis(x1, LMprob, opt1)

        if opt2.algorithm == :MPSO_full_state
            x2 = [f2(A);w]
        else
            x2 = f2(A)
        end

        rf2 = _singleAttAnalysis(x2, LMprob, opt2)


        resultsFull1[i] = LMoptimizationResults(rF1, x1, LMprob, opt1)

        resultsFull2[i] = LMoptimizationResults(rF2, x2, LMprob, opt2)
    end

    return (resultsFull1, resultsFull2)
end

end # module
