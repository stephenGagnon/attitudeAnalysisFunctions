module attitudeAnalysisFunctions

using attitudeFunctions
using lightCurveOptimization
using Distributed
using SharedArrays
using Infiltrator

export attitudeOptimizer, randomAttAnalysis

function attitudeOptimizer(trueAttitude, params :: Union{PSO_parameters,GB_parameters},
        options = optimizationOptions() :: optimizationOptions, a = 1.0 , f = 1.0;
        object = "simple satellite", scenario = "simple scenario",
        initialParticleDistribution = 0)

    if cmp(object,"simple satellite") == 0
        (sat, satFull) = simpleSatellite(vectorized = options.vectorizeCost)
    else
        error("Please Provide valid object specifiction. Options are:
        'simple satellite'")
    end

    if cmp(scenario,"simple scenario") == 0
        scen = simpleScenario(vectorized = options.vectorizeCost)
    else
        error("Please Provide valid object specifiction. Options are:
        'simple scenario'")
    end

    if typeof(trueAttitude) == Array{Float64,2}
        A = trueAttitude
    elseif typeof(trueAttitude) == Union{MRP,GRP,quaternion,DCM}
        A = any2A(trueAttitude)
    elseif (typeof(trueAttitude) == Array{Float64,1}) & (length(trueAttitude) == 4)
        A = q2A(trueAttitude)
    elseif (typeof(trueAttitude) == Array{Float64,1}) & (length(trueAttitude) == 3)
        A = p2A(trueAttitude,a,f)
    else
        error("please provide a valid true attitude")
    end

    costFunc = costFuncGen(sat,scen,A,options,a,f)


    if options.algorithm == "MPSO"

        if options.initMethod == "random"
            xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
        elseif coptions.initMethod == "specified"
            xinit = initialParticleDistribution
        else
            error("Please provide valid particle initialization method")
        end
        results = MPSO_cluster(xinit,costFunc,params)
    elseif options.algorithm == "PSO_cluster"

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
    elseif options.algorithm == "SPQ"

    end

    return optimizationResults(results, sat, satFull, scen, params, trueAttitude, options)
end

function randomAttAnalysis(N :: Int64, params :: PSO_parameters,
        options = optimizationOptions() :: optimizationOptions, a = 1.0 , f = 1.0;
        object = "simple satellite", scenario = "simple scenario")

    if cmp(object,"simple satellite") == 0
        (sat, satFull) = simpleSatellite(vectorized = options.vectorizeCost)
    else
        error("Please Provide valid object specifiction. Options are:
        'simple satellite'")
    end

    if cmp(scenario,"simple scenario") == 0
        scen = simpleScenario(vectorized = options.vectorizeCost)
    else
        error("Please Provide valid object specifiction. Options are:
        'simple scenario'")
    end

    trueAttitudes = randomAtt(N,DCM,options.vectorizeOptimization)

    # resultsFull = SharedArray{PSO_results,1}(undef,N)
    resultsFull = Array{PSO_results,1}(undef,N)

    for i = 1:N

        A = trueAttitudes[i]
        costFunc = costFuncGen(sat,scen,A,options,a,f)

        if options.algorithm == "MPSO"

            if options.initMethod == "random"
                xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
            elseif coptions.initMethod == "specified"
                xinit = initialParticleDistribution
            else
                error("Please provide valid particle initialization method")
            end
            results = MPSO_cluster(xinit,costFunc,params)

        elseif options.algorithm == "PSO_cluster"

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

        end

        resultsFull[i] = results
    end

    return optimizationResults(resultsFull, sat, satFull, scen, params, trueAttitudes, options)
end

end # module
