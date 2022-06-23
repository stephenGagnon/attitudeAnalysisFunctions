# function processScenarioInputs(object, scenario, options)
#
#     if object[1] == :simple
#         sat, satFull = simpleSatellite(vectorized = options.vectorize)
#     elseif object[1] == :modified
#         sat, satFull = customSatellite(object[2], vectorized = options.vectorize)
#     elseif object[1] == :custom
#         (sat, satFull) = object[2]
#     else
#         error("Please Provide valid object specifiction. Options are:
#         :simple, :custom")
#     end
#
#     if scenario[1] == :simple
#         scen = simpleScenario(vectorized = options.vectorizeCost)
#     elseif scenario[1] == :modified
#         scen = customScenario(scenario[2],vectorized = options.vectorize)
#     elseif object[1] == :custom
#         scen = scenario[2]
#     else
#         error("Please Provide valid object specifiction. Options are:
#         'simple scenario'")
#     end
#
#     return sat, satFull, scen
# end

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

# costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
# clusterFunc = (x,N,cl) -> cl[:] = (assignments(kmeans(x,N)))
#
# if options.initMethod == :random
#     xinit = randomAtt(1,options.Parameterization)
# elseif options.initMethod == :specified
#     xinit = options.initVals
# else
#     error("Please provide valid particle initialization method")
# end
# results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
# elseif options.algorithm == :MPSO_VGC
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#
#     if (options.Parameterization == MRP) | (options.Parameterization == GRP)
#         rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
#         dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
#     elseif options.Parameterization == quaternion
#         rotFunc = qRotate :: Function
#         dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
#     else
#         error("Please provide a valid attitude representation type. Options are:
#         'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
#         or 'quaternion' ")
#     end
#
#     visGroups = Array{visibilityGroup,1}(undef,0)
#     clusterFunc = ((x :: ArrayOfVecs,N :: Int64, ind :: Vector{Int64})-> visGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{visibilityGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)) :: Function
#
#     # visGroups = Array{sunVisGroup,1}(undef,0)
#     # clusterFunc = ((x :: ArrayOfVecs,N :: Int64, ind :: Vector{Int64})-> sunVisGroupClustering(x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}, visGroups :: Vector{sunVisGroup}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)) :: Function
#
#
#     if options.initMethod == :random
#         xinit = randomAtt(1,options.Parameterization)
#     elseif options.initMethod == :specified
#         xinit = options.initVals
#     else
#         error("Please provide valid particle initialization method")
#     end
#     results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
#
# elseif options.algorithm == :MPSO_NVC
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#
#     if (options.Parameterization == MRP) | (options.Parameterization == GRP)
#         rotFunc = ((A,v) -> p2A(A,a,f)*v) :: Function
#         dDotFunc = ((v1,v2,att) -> -dDotdp(v1,v2,-att))
#     elseif options.Parameterization == quaternion
#         rotFunc = qRotate :: Function
#         dDotFunc = ((v1,v2,att) -> qinv(dDotdq(v1,v2,qinv(att))))
#     else
#         error("Please provide a valid attitude representation type. Options are:
#         'MRP' (modified Rodrigues parameters), 'GRP' (generalized Rodrigues parameters),
#         or 'quaternion' ")
#     end
#
#     clusterFunc = (x :: ArrayOfVecs, N :: Int64, ind :: Vector{Int64}) -> normVecClustering(x :: ArrayOfVecs, ind :: Vector{Int64}, sat :: targetObject, scen :: spaceScenario, rotFunc :: Function)
#
#     if options.initMethod == :random
#         xinit = randomAtt(1,options.Parameterization)
#     elseif options.initMethod == :specified
#         xinit = options.initVals
#     else
#         error("Please provide valid particle initialization method")
#     end
#     results = MPSO_cluster(xinit,costFunc,clusterFunc,params)
#
# elseif options.algorithm == :PSO_cluster
#
#     costFunc = costFuncGenPSO(sat,scen,Atrue,options,a,f)
#     if options.initMethod == :random
#         xinit = randomAtt(1,options.Parameterization)
#     elseif options.initMethod == :specified
#         xinit = options.initVals
#     else
#         error("Please provide valid particle initialization method")
#     end
#     results = PSO_cluster(xinit,costFunc,params)
#     if options.vectorizeOptimization
#         results = Convert_PSO_results(results,options.Parameterization,a,f)
#     end
