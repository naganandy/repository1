import os, pickle, inspect, numpy as np, scipy.io
from tqdm import tqdm


'''
returns
1) hyperlinks: full incidence matrix of the hypergraph
2) candidates: incidence-like matrix for all the candidates

if data is modelled using a directed hypergraph, additionally return
3) directed hyperlinks
4) directed candidates 
'''
def load(args):
	
	dataset = {}
	
	dataset['hyperlinks'] = loadFile(args, "hyperlinks")
	dataset['candidates'] = loadFile(args, "candidates")

	if args.data == "reactionsD":
		dataset['dhyperlinks'] = loadFile(args, "dhyperlinks")
		dataset['dcandidates'] = loadFile(args, "dcandidates")
	
	CHL = sanityCheck(dataset['hyperlinks'], dataset['candidates'])   # check if all hyperlinks are candidates (necessary)
	

	return dataset






# (sanity) check if each hyperlink in the data is a candidate
def sanityCheck(hyperlinks, candidates):
    E = hyperlinks.shape[1] # number of hyperlinks
    CHL = len(candidates[0]) # number of candidates

    print("sanity check: checking if all hyperlinks are candidates...", end='')
    for column in range(E):
        if hyperlinks[:,column] in candidates.T:
            continue
        else:
            print("\n error in hyperlinks and/or candidates\n ")
            CHL = -1
            break
    if CHL != -1:
    	print("done!")
    return CHL







'''
load the dataset file
returns:
matrix of content
'''
def loadFile(args, content):
	currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	file = os.path.join(currentdir, args.data, args.dataset, args.dataset + "." + content + ".pkl")
	
	matrix = None
	with open(file, 'rb') as handle:
		matrix = pickle.load(handle)
	return matrix







# load datasets such as recipes and metabolic reactions
'''
returns:
1) graph (either the untouched hypergraph or the clique expansion of it)
2) features on the nodes
3) labels on the nodes
'''
def preProcess(dataset, args):

	candidates = dataset['candidates']
	hyperlinks = dataset['hyperlinks']

	dataset['graph'] = getGraph(candidates)
	dataset['features'] = getFeatures(candidates, args)
	dataset['labels'] = getLabels(hyperlinks, candidates)

	return dataset









# get the primal or the dual hypergraph from the candidates
'''
returns: primal/dual hypergraph untouched or approximated using clique expansion
'''
def getGraph(candidates):
	graph = Dual(candidates)
	graph = clique_expansion(graph, n = candidates.shape[1])
	return graph






# get the features of the nodes of the graph
'''
returns: np.ndarray of either 
random Gaussian features
OR
node2vec features
'''
def getFeatures(candidates, args):
	features, V = [], 0
	n = candidates.shape[1]

	# random Gaussian features
	print("node feature initialisation: ", end = '')
	if args.data == 'coauthorship':
		content = loadFile(args, "features")
		features = []
		for feat in content:
			features.append(np.array(feat))
		features = np.array(features).astype(np.float)
	else:
		num_features = args.numFeatures
		mean = [0]*(num_features)
		cov = list(np.eye(num_features))
		
		print("generating random",num_features,"-dimensional Gaussian features...", end = '')
		for i in range(n):
			feat = np.random.multivariate_normal(mean, cov).T
			features.append(feat)
	print("done!")
	return features








# get the labels on the nodes
'''
returns: one-hot labels on the nodes
'''
def getLabels(hyperlinks, candidates):
	n = candidates.shape[1]
	labels = [[0,1]]*n
	
	searchList = list(hyperlinks.T)
	print("label assignment: giving one-hot labels to nodes...", end = '')
	for column in range(hyperlinks.shape[1]):
		labels[column] = [1,0]
	print("done!")
	return labels










# get the dual of the hypergraph
def Dual(candidates):
	'''
	dual: dictionary of hyperlinks (key is a hyperlink id and value is the corresponding set of hypernodes)
	n: number of hypernodes in the primal
	m: number of candidate hyperlinks in the primal
	'''
	dual, n, m = {}, candidates.shape[0], candidates.shape[1] 

	print("graph construction: constructing the dual hypergraph of", m, "hypernodes and", n, "hyperlinks...", end='')
	# each hypernode is a hyperlink in the dual and vice versa
	for i in range(n): 
		for j in range(m):
			if candidates[i][j] != 0:
				key = str(i)
				if key not in dual.keys():
					dual[key] = set()
				dual[key].add(j)
	print("done!")
	return dual










'''
connect every pair of nodes in each hyperedge by an edge
a hyperdge of size s is approximated by an s-clique
'''
def clique_expansion(hypergraph, n):
    ce = np.eye(n)
    print("graph construction: approximating the hypergraph by its clique expansion...", end='')
    for key in hypergraph.keys():
        e = list(hypergraph[key])
        l = len(e)
        for i in range(l):
            for j in range(i+1,l):
                I, J = e[i], e[j]
                ce[I][J], ce[J][I] = 1, 1
    print("done!")
    return ce


