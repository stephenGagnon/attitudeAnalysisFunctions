module attitudeAnalysisFunctions

using attitudeFunctions
using LinearAlgebra
using Random
using lightCurveOptimization
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

export randomAttAnalysis, generateLMData, findLevelSets, GBcleanup, monteCarloAttAnalysis, singleAttAnalysis, analyzeRandomAttitudeConvergence, optimizationComparison

function monteCarloAttAnalysis(Atrue :: anyAttitude, N :: Int64, params :: Union{PSO_parameters,GB_parameters}, options = optimizationOptions() :: optimizationOptions; a = 1.0 , f = 1.0, object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    if any(options.algorithm .== [:MPSO,:PSO_cluster,:MPSO_AVC])
        resultsFull = Array{PSO_results,1}(undef,N)
    elseif any(options.algorithm .== [:LD_SLSQP])
        resultsFull = Array{GB_results,1}(undef,N)
    else
        @infiltrate
    end

    for i = 1:N
        resultsFull[i] = singleAttAnalysis(Atrue, params, options, a = 1.0 , f = 1.0, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup)
    end

    return optimizationResults(resultsFull, sat, satFull, scen, params, trueAttitudes, options)
end

function singleAttAnalysis(Atrue :: anyAttitude, params :: Union{PSO_parameters,GB_parameters}, options = optimizationOptions() :: optimizationOptions; a = 1.0 , f = 1.0, object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    # if object[1] == :simple
    #     (sat, satFull) = simpleSatellite(vectorized = options.vectorizeCost)
    # elseif object[1] == :modified
    #     (sat, satFull) = customSatellite(object[2], vectorized = options.vectorizeCost)
    # elseif object[1] == :custom
    #     (sat, satFull) = object[2]
    # else
    #     error("Please Provide valid object specifiction. Options are:
    #     :simple, :custom")
    # end
    #
    # if scenario[1] == :simple
    #     scen = simpleScenario(vectorized = options.vectorizeCost)
    # elseif scenario[1] == :modified
    #     scen = customScenario(scenario[2],vectorized = options.vectorizeCost)
    # elseif object[1] == :custom
    #     scen = object[2]
    # else
    #     error("Please Provide valid object specifiction. Options are:
    #     'simple scenario'")
    # end
    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    if options.algorithm == :MPSO

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

        if options.initMethod == "random"
            xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
        elseif options.initMethod == "specified"
            xinit = initialParticleDistribution
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
    elseif options.algorithm == :MPSO_AVC

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))

        if options.initMethod == "random"
            xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
        elseif coptions.initMethod == "specified"
            xinit = initialParticleDistribution
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


        if options.initMethod == "random"
            xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
        elseif coptions.initMethod == "specified"
            xinit = initialParticleDistribution
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
    elseif options.algorithm == :PSO_cluster

        costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
        if options.initMethod == "random"
            xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
        elseif coptions.initMethod == "specified"
            xinit = initialParticleDistribution
        else
            error("Please provide valid particle initialization method")
        end
        results = PSO_cluster(xinit,costFunc,params)
        if options.vectorizeOptimization
            results = Convert_PSO_results(results,options.Parameterization,a,f)
        end
    elseif any(options.algorithm .== [:LD_SLSQP])

        if options.initMethod == "random"
            xinit = randomAtt(1,options.Parameterization)
        elseif coptions.initMethod == "specified"
            xinit = initialParticleDistribution
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

            if gbf(minx,[0;0;0.0]) != minf
                @infiltrate
            end
            if any(isnan.(minx)) | any(isnan.(p2q(minx)))
                @infiltrate
            end
            if abs(norm(p2q(minx))-1) > .000001
                @infiltrate
            end
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

function randomAttAnalysis(N :: Int64, params :: Union{PSO_parameters,GB_parameters}, options = optimizationOptions() :: optimizationOptions; object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = false)

    sat, satFull, scen = processScenarioInputs(object, scenario, options)

    trueAttitudes = randomAtt(N,DCM,options.vectorizeOptimization)

    # resultsFull = SharedArray{PSO_results,1}(undef,N)
    if any(options.algorithm .== [:MPSO,:MPSO_VGC,:PSO_cluster,:MPSO_AVC])
        resultsFull = Array{PSO_results,1}(undef,N)
    elseif any(options.algorithm .== [:LD_SLSQP])
        resultsFull = Array{GB_results,1}(undef,N)
    else
        @infiltrate
    end


    for i = 1:N

        if N>1
            A = trueAttitudes[i]
        elseif N == 1
            A = trueAttitudes
        end

        resultsFull[i] = singleAttAnalysis(A, params, options, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup)

    end

    return optimizationResults(resultsFull, sat, satFull, scen, params, trueAttitudes, options)
end

function optimizationComparison(opt1 :: optimizationOptions, opt2 :: optimizationOptions, params1 :: Union{PSO_parameters,GB_parameters}, params2 :: Union{PSO_parameters,GB_parameters}, N :: Int64; object = (:simple,nothing), scenario = (:simple,nothing), GBcleanup = (false,false))

    sat, satFull, scen = processScenarioInputs(object, scenario, opt1)

    trueAttitudes = randomAtt(N,DCM,opt1.vectorizeOptimization)

    # resultsFull = SharedArray{PSO_results,1}(undef,N)
    if any(opt1.algorithm .== [:MPSO,:MPSO_VGC,:PSO_cluster,:MPSO_AVC])
        resultsFull1 = Array{PSO_results,1}(undef,N)
    elseif any(options.algorithm .== [:LD_SLSQP])
        resultsFull1 = Array{GB_results,1}(undef,N)
    else
        @infiltrate
    end

    if any(opt2.algorithm .== [:MPSO,:MPSO_VGC,:PSO_cluster,:MPSO_AVC])
        resultsFull2 = Array{PSO_results,1}(undef,N)
    elseif any(options.algorithm .== [:LD_SLSQP])
        resultsFull2 = Array{GB_results,1}(undef,N)
    else
        @infiltrate
    end


    for i = 1:N

        if N>1
            A = trueAttitudes[i]
        elseif N == 1
            A = trueAttitudes
        end

        resultsFull1[i] = singleAttAnalysis(A, params1, opt1, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup[1])

        resultsFull2[i] = singleAttAnalysis(A, params2, opt2, object = (:custom, (sat,satFull)), scenario = (:custom,scen), GBcleanup = GBcleanup[2])
    end

    return (optimizationResults(resultsFull1, sat, satFull, scen, params1, trueAttitudes, opt1), optimizationResults(resultsFull2, sat, satFull, scen, params2, trueAttitudes, opt2))
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
