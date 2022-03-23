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

import Clustering: kmeans, kmedoids, assignments

const Vec{T<:Number} = AbstractArray{T,1}
const Mat{T<:Number} = AbstractArray{T,2}
const ArrayOfVecs{T<:Number} = Array{V,1} where V <: Vec
const ArrayofMats{T<:Number} = Array{M,1} where M <: Mat
const MatOrVecs = Union{Mat,ArrayOfVecs}
const MatOrVec = Union{Mat,Vec}
const anyAttitude = Union{Mat,Vec,DCM,MRP,GRP,quaternion}
const Num = N where N <: Number

include("utilities.jl")

export randomAttAnalysis, generateLMData, findLevelSets, GBcleanup, monteCarloAttAnalysis, singleAttAnalysis, analyzeRandomAttitudeConvergence, optimizationComparison, multiAttAnalysis

function monteCarloAttAnalysis(Atrue :: anyAttitude, N :: Int64; params = PSO_parameters(), options = optimizationOptions(), a = 1.0 , f = 1.0, object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    resultsFull = Array{optimizationResults,1}(undef,N)

    for i = 1:N
         temp = singleAttAnalysis(Atrue, params = params, otpions = options, a = 1.0 , f = 1.0, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup)

        resultsFull[i] = optimizationResults(temp, sat, satFull, scen, params, Atrue, options)
    end

    return resultsFull
end

function _singleAttAnalysis(Atrue :: anyAttitude; params = PSO_parameters(), options = optimizationOptions(), a = 1.0 , f = 1.0, object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    if options.algorithm == :MPSO

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,clusterFunc,params)

    elseif options.algorithm == :MPSO_AVC

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_AVC(xinit,costFunc,clusterFunc,params)

    elseif options.algorithm == :MPSO_VGC

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)

        if (options.Parameterization == MRP) | (options.Parameterization == GRP)
            rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
            dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
        elseif options.Parameterization == quaternion
            rotFunc = qRotate :: Function
            dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
        else
            error("Please provide a valid attitude representation type. Options are:
            'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
            or 'quaternion' ")
        end

        visGroups = Array{visibilityGroup,1}(undef,0)
        clusterFunc = ((x :: ArrayOfVecs,N :: Int64, ind :: Vector{Int64})-> visGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{visibilityGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)) :: Function

        # visGroups = Array{sunVisGroup,1}(undef,0)
        # clusterFunc = ((x :: ArrayOfVecs,N :: Int64, ind :: Vector{Int64})-> sunVisGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{sunVisGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)) :: Function


        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,clusterFunc,params)

    elseif options.algorithm == :MPSO_NVC

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)

        if (options.Parameterization == MRP) | (options.Parameterization == GRP)
            rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
            dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
        elseif options.Parameterization == quaternion
            rotFunc = qRotate :: Function
            dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
        else
            error("Please provide a valid attitude representation type. Options are:
            'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
            or 'quaternion' ")
        end

        clusterFunc = (x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}) -> normVecClustering(x :: ArrayOfVecs, ind :: Vector{Int64}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)

        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,clusterFunc,params)

    elseif options.algorithm == :PSO_cluster

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        results = PSO_cluster(xinit,costFunc,params)
        if options.vectorizeOptimization
            results = Convert_PSO_results(results,options.Parameterization,a,f)
        end

    elseif any(options.algorithm .== [:LD_SLSQP])
        if options.initMethod == :random
            xinit = randomAtt(1,options.Parameterization)
        elseif options.initMethod == :specified
            xinit = options.initVals
        else
            error("Please provide valid particle initialization method")
        end
        if any(abs.(xinit) .> 1)
            xinit = sMRP(xinit)
        end
        opt = Opt(options.algorithm,3)
        opt.min_objective = costFuncGenNLopt(sat,scen,Atrue,options,a,f)
        opt.lower_bounds = [-1;-1;-1]
        opt.upper_bounds = [1;1;1]
        opt.maxeval = params.maxeval
        opt.maxtime = params.maxtime
        (minf, minx, ret) = optimize(opt,xinit)
        results = GB_results(minf,minx,ret)
    end

    if GBcleanup
        opti = optimizationOptions(Parameterization = MRP, algorithm = :LD_SLSQP)
        opt = Opt(opti.algorithm,3)
        gbf = costFuncGenNLopt(sat,scen,Atrue,options,1.0,1.0)
        opt.min_objective = gbf
        opt.lower_bounds = [-1;-1;-1]
        opt.upper_bounds = [1;1;1]
        opt.maxeval = 1000
        opt.maxtime = 1

        cluster_xopt = results.clusterxOptHist[end]
        cluster_fopt = results.clusterfOptHist[end]

        cleaned_xopt = similar(cluster_xopt)
        cleaned_fopt = similar(cluster_fopt)

        for i = 1:size(cluster_xopt,1)
            if options.algorithm == :MPSO
                xinit = q2p(cluster_xopt[:,i])
            else
                xinit = cluster_xopt[:,i]
            end
            if norm(xinit) > 1
                xinit = -xinit/dot(xinit,xinit)
            end
            (minf, minx, ret) = optimize(opt,xinit)

            # if gbf(minx,[0;0;0.0]) != minf
            #     @infiltrate
            # end
            # if any(isnan.(minx)) | any(isnan.(p2q(minx)))
            #     @infiltrate
            # end
            # if abs(norm(p2q(minx))-1) > .000001
            #     @infiltrate
            # end
            if options.algorithm == :MPSO
                cleaned_xopt[:,i] = p2q(minx)
            else
                cleaned_xopt[:,i] = minx
            end

            cleaned_fopt[i] = minf
        end

        (fopt,minind) = findmin(cleaned_fopt)
        resultsOut = PSO_results(results.xHist, results.fHist, results.xOptHist,
         results.fOptHist, results.clusterxOptHist,results.clusterfOptHist,
         cleaned_xopt[:,minind], fopt)
    else
        resultsOut = results
    end


    return resultsOut
end

function singleAttAnalysis(Atrue :: anyAttitude; params = PSO_parameters(), options = optimizationOptions(), a = 1.0 , f = 1.0, object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    results = _singleAttAnalysis(Atrue; params = params, options = options, a = a , f = f, object = object, scenario = scenario, GBcleanup = GBcleanup)

    return optimizationResults(results, sat, satFull, scen, params, Atrue, options)
end

function randomAttAnalysis(N :: Int64; params = PSO_parameters(), options = optimizationOptions(), object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    trueAttitudes = randomAtt(N,DCM,options.vectorizeOptimization)

    return multiAttAnalysis(N, trueAttitudes, params = params, options = options, object = (:custom,(sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup)
end

function multiAttAnalysis(N :: Int64, Attitudes; params = PSO_parameters(), options = optimizationOptions(), object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    resultsFull = Array{optimizationResults,1}(undef,N)

    for i = 1:N

        if N>1
            A = Attitudes[i]
        elseif N == 1
            A = Attitudes
        end

        resultsFullTemp = _singleAttAnalysis(A, params = params, options = options, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup)

        resultsFull[i] = optimizationResults(resultsFullTemp, sat, satFull, scen, params, A, options)

    end

    return resultsFull
end

function optimizationComparison(opt1 :: optimizationOptions, opt2 :: optimizationOptions, params1 :: Union{PSO_parameters,GB_parameters}, params2 :: Union{PSO_parameters,GB_parameters}, N :: Int64; object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = (false,false))

    sat, satFull, scen = processScenarioInputs(object, scenario, opt1)

    trueAttitudes = randomAtt(N,DCM,opt1.vectorizeOptimization)

    resultsFull1 = Array{optimizationResults,1}(undef,N)
    resultsFull2 = Array{optimizationResults,1}(undef,N)

    for i = 1:N

        if N>1
            A = trueAttitudes[i]
        elseif N == 1
            A = trueAttitudes
        end

        rF1 = _singleAttAnalysis(A, params = params1, options = opt1, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup[1])

        rf2 = _singleAttAnalysis(A, params = params2, options = opt2, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup[2])

        resultsFull1[i] = optimizationResults(rF1, sat, satFull, scen, params1, trueAttitudes[i], opt1)

        resultsFull2[i] = optimizationResults(rF2, sat, satFull, scen, params2, trueAttitudes[i], opt2)
    end

    return (resultsFull1, resultsFull2)
end



# if options.algorithm == :MPSO
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#     clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))
#
#     if options.initMethod == "random"
#         xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
#     elseif coptions.initMethod == "specified"
#         xinit = initialParticleDistribution
#     else
#         error("Please provide valid particle initialization method")
#     end
#
#     results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
# elseif options.algorithm == :MPSO_AVC
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#     clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))
#
#     if options.initMethod == "random"
#         xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
#     elseif coptions.initMethod == "specified"
#         xinit = initialParticleDistribution
#     else
#         error("Please provide valid particle initialization method")
#     end
#
#     results = MPSO_AVC(xinit,costFunc,clusterFunc,params)
# elseif options.algorithm == :PSO_cluster
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#
#     if options.initMethod == "random"
#         xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
#     elseif coptions.initMethod == "specified"
#         xinit = initialParticleDistribution
#     else
#         error("Please provide valid particle initialization method")
#     end
#
#     results = PSO_cluster(xinit,costFunc,params)
#
#     if options.vectorizeOptimization
#         results = Convert_PSO_results(results,options.Parameterization,a,f)
#     end
# elseif any(options.algorithm .== [:LD_SLSQP])
#
#     if options.initMethod == "random"
#         xinit = randomAtt(1,options.Parameterization)
#     elseif coptions.initMethod == "specified"
#         xinit = initialParticleDistribution
#     else
#         error("Please provide valid particle initialization method")
#     end
#
#     if any(abs.(xinit) .> 1)
#         xinit = sMRP(xinit)
#     end
#
#     opt = Opt(options.algorithm,3)
#     opt.min_objective = costFuncGenNLopt(sat,scen,Atrue,options,a,f)
#     opt.lower_bounds = [-1;-1;-1]
#     opt.upper_bounds = [1;1;1]
#     opt.maxeval = params.maxeval
#     opt.maxtime = params.maxtime
#     (minf, minx, ret) = optimize(opt,xinit)
#
#     results = GB_results(minf,minx,ret)
# end
end # module
