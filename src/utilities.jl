function GBcleanup(results :: optimizationResults)

    options = optimizationOptions(Parameterization = MRP, algorithm = :LD_SLSQP)
    opt = Opt(options.algorithm,3)
    f = costFuncGenNLopt(results.object,results.scenario,results.trueAttitude,options,1.0,1.0)
    opt.min_objective = f
    opt.lower_bounds = [-1;-1;-1]
    opt.upper_bounds = [1;1;1]
    opt.maxeval = 1000
    opt.maxtime = 1

    Pr = results.results
    cluster_xopt = Pr.clusterxOptHist[end]
    cluster_fopt = Pr.clusterfOptHist[end]

    cleaned_xopt = similar(cluster_xopt)
    cleaned_fopt = similar(cluster_fopt)

    for i = 1:size(cluster_xopt,1)
        if options.algorithm == :MPSO
            xinit = q2p(cluster_xopt[i,:])
        else
            xinit = cluster_xopt[i,:]
        end

        if norm(xinit) > 1
            xinit = -xinit/dot(xinit,xinit)
        end
        (minf, minx, ret) = optimize(opt,xinit)

        # if f(minx,[0;0;0.0]) != minf
        #     # @infiltrate
        # end
        # if any(isnan.(minx)) | any(isnan.(p2q(minx)))
        #     @infiltrate
        # end
        # if abs(norm(p2q(minx))-1) > .000001
        #     @infiltrate
        # end
        cleaned_xopt[:,i] = p2q(minx)
        cleaned_fopt[i] = minf
    end

    (fopt,minind) = findmin(cleaned_fopt)
    Prc = PSO_results(Pr.xHist, Pr.fHist, Pr.xOptHist, Pr.fOptHist, Pr.clusterxOptHist,
     Pr.clusterfOptHist, cleaned_xopt[:,minind], fopt)

    return optimizationResults(Prc,results.object,results.objectFullData,results.scenario,
    results.PSO_params,results.trueAttitude,results.options)
end

function generateLMData(N;Atrue = nothing)

    att = randomAttPar(N,MRP)
    (sat,_, scen) = simpleScenarioGenerator(vectorized = false)

    if Atrue != nothing
        att[:,1] = Atrue
    end
    obsNo = scen.obsNo

    LM = Matrix{Float64}(undef,obsNo,N)

    Threads.@threads for i = 1:N
        LM[:,i] = Fobs(att[:,i],sat,scen)
    end

    jldsave("data/LMData.jld2";att,LM)
end

function findLevelSets(loadFileName,saveFileName,LMt,tol)

    data = load(loadFileName)
    att = get(data,"att",nothing)
    LM = get(data,"LM",nothing)

    N = size(LM,2)
    obsNo = size(LM,1)

    isinlevelset = Matrix{Bool}(undef,obsNo,N)

    Threads.@threads for i = 1:N
        for j = 1:obsNo
            if abs(LM[j,i]-LMt[j]) < tol
                isinlevelset[j,i] = true
            else
                isinlevelset[j,i] = false
            end
        end
    end

    levelsets = Vector{Any}(undef,obsNo)
    levelsetLM = Vector{Any}(undef,obsNo)
    for j = 1:obsNo
        levelsets[j] = att[:,isinlevelset[j,:]]
        levelsetLM[j] = LM[j,isinlevelset[j,:]]
    end

    jldsave(saveFileName;levelsets,levelsetLM)
    # return levelsets, levelsetLM
end

function processScenarioInputs(object, scenario, options)

    if object[1] == :simple
        sat, satFull = simpleSatellite(vectorized = options.vectorizeCost)
    elseif object[1] == :modified
        sat, satFull = customSatellite(object[2], vectorized = options.vectorizeCost)
    elseif object[1] == :custom
        (sat, satFull) = object[2]
    else
        error("Please Provide valid object specifiction. Options are:
        :simple, :custom")
    end

    if scenario[1] == :simple
        scen = simpleScenario(vectorized = options.vectorizeCost)
    elseif scenario[1] == :modified
        scen = customScenario(scenario[2],vectorized = options.vectorizeCost)
    elseif object[1] == :custom
        scen = scenario[2]
    else
        error("Please Provide valid object specifiction. Options are:
        'simple scenario'")
    end

    return sat, satFull, scen
end

function analyzeRandomAttitudeConvergence(results)

    (optConv, optErrAng, clConv, clErrAng) = checkConvergence(results)
    N = length(optConv)
    #
    convFrac = length(findall(optConv))/N
    clConvFrac = length(findall(clConv))/N

    return convFrac,clConvFrac
end
