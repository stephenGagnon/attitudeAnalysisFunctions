module attitudeAnalysisFunctions

using lightCurveOptimization
using attitudeFunctions

export attitudePSO_Main randomAttAnalysis

function attitudePSO_Main(trueAttitude, params :: PSO_parameters,
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

    if cmp(options.initMethod,"random") == 0
        xinit = randomAtt(params.N,options.Parameterization,options.vectorizeOptimization)
    elseif cmp(options.initMethod,"specified") == 0
        xinit = initialParticleDistribution
    else
        error("Please provide valid particle initialization method")
    end

    if options.useMPSO
        results = MPSO_cluster(xinit,costFunc,params)
    else
        results = PSO_cluster(xinit,costFunc,params)
        if options.customTypes
            results = Convert_PSO_results(results,options.Parameterization,a,f)
        end
    end

    return optimizationResults(results, sat, satFull, scen, params, trueAttitude, options)
end

function randomAttAnalysis(N :: Int64, attType, params :: PSO_parameters,
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



end

end # module
