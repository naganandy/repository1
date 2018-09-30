import random, os, inspect, torch
from scipy import spatial
from torch.autograd import Variable


'''
get train-test split for data
'''
def TrainTest(dataset, args):
    train, test = {}, {}
    n = len(dataset['features'])
    labels = dataset['labels']

    if args.data == "reactionsD":
        directedHyperlinks = dataset['dhyperlinks']
        E, pospairs = getPositiveDirectedandUndirectedHyperlinks(args, directedHyperlinks)
        
        directedCandidates = dataset['dcandidates']
        negpairs, labels = addPULHyperlinksDirected(labels, pospairs, directedCandidates, n, args)

        negIndices = set()
        for pair in negpairs:
            negIndices.add(pair[0])
            negIndices.add(pair[1])
        F = list(negIndices)

        train['trainD'] = pospairs + negpairs
        test['testD'] =  list(set(directedCandidates) - set(pospairs) - set(negpairs))

        directions = Variable(torch.LongTensor(len(train['trainD'])))
        directions[0:len(pospairs)] = 0
        directions[len(pospairs):] = 1
        train['directions'] = directions

        directions = Variable(torch.LongTensor(len(test['testD'])))
        for i,pair in enumerate(test['testD']):
            if pair in directedHyperlinks:
                directions[i] = 0
            else:
                directions[i] = 1
        test['directions'] = directions
        args.numMissing = len(dataset['dhyperlinks']) - len(E)
    else:
        positives = getHyperlinkIndicesCandidates(labels)
        E = random.sample(positives, int(round(1.0-args.missingFrac, 2)*len(positives)))
        args.numMissing = len(positives) - len(E)

        F, labels = addPULHyperlinks(labels, E, n, args)  # PUL: positive unlabelled learning
    train['trainU'] = E + F
    test['testU'] = list(set(list(range(n))) - set(train['trainU']))

    return train, test






'''
PUL: positive unlabelled learning 
select a certain specified number of most dissimilar nodes with 
respect to average node2vec similarity and label each negative 
'''
def addPULHyperlinks(labels, E, n, args):
    embeddings = node2vec(args)
    
    similarity = {}
    for u in range(n):
        if u not in E:
            similarity[str(u)] = compute_average_similarity(u, embeddings, E)
    
    sorted_indices = sorted(similarity, key=lambda k: similarity[k])
    sorted_indices = [int(i) for i in sorted_indices]
    F = sorted_indices[0: len(E)]

    for neg in F:
        labels[neg] = [0,1]
    return F, labels






def concatenateEmbeddings(pospairs, embeddings):
    emb = []
    for pair in pospairs:
        i,j = pair[0],pair[1]
        catemb = embeddings[str(i)] + embeddings[str(j)]#np.concatenate(np.array(embeddings[str(i)], embeddings[str(j)]))
        emb.append(catemb)
    return emb


'''
PUL: positive unlabeled learning 
select the k most dissimilar nodes with respect to average node2vec similarity and label each negative 
'''
def addPULHyperlinksDirected(labels, pospairs, directedCandidates, n, args):
    embeddings = node2vec(args)

    catembeddings = concatenateEmbeddings(pospairs, embeddings)
    
    similarity = {}
    for pair in directedCandidates:
        if pair not in pospairs:
            i,j = pair
            emb_pair = embeddings[str(i)] + embeddings[str(j)]#np.concatenate(embeddings[str(i)], embeddings[str(j)])
            similarity[pair] = compute_average_similarity_pair(emb_pair, catembeddings)
    
    sorted_pairs = sorted(similarity, key=lambda k: similarity[k])
    # sorted_indices = [int(i) for i in sorted_indices]
    negpairs = sorted_pairs[0: len(pospairs)]

    #gt = Variable(torch.LongTensor(len(pairs)))
    
    for k, pair in enumerate(negpairs):
        i,j = pair
        labels[i] = [0,1]
        labels[j] = [0,1]
    return negpairs, labels



'''
compute average similarity with respect to the positive indices
'''

def compute_average_similarity_pair(emb_pair, catembeddings):
    sum = 0
    for cat_emb in catembeddings:
        result = 1 - spatial.distance.cosine(emb_pair, cat_emb)
        sum = sum + result
    return sum




'''
compute average similarity with respect to the positive indices
'''
def compute_average_similarity(node, embeddings, positives):
    sum = 0
    emb = embeddings[str(node)]
    for pos in positives:
        embP = embeddings[str(pos)]
        result = 1 - spatial.distance.cosine(emb, embP)
        sum = sum + result
    return sum






# return node2vec embeddings of nodes of the graph 
def node2vec(args):

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    file = os.path.join(currentdir, args.data, args.dataset, args.dataset + ".dual.emd")
    with open(file, "r") as f:
        lines = f.readlines()
        first = lines[0]

        node_embeddings = {}
        for i in range(1, len(lines)):
            line = lines[i]
            node_id = int(line.strip().split()[0])

            node_embedding = list(line.strip().split()[1:])
            node_embedding = [float(v) for v in node_embedding]

            node_embeddings[str(node_id)] = node_embedding
    return node_embeddings




'''
return a list of indices of hyperlinks in candidates
i.e. a list of indices i for which labels[i] is positive
'''
def getHyperlinkIndicesCandidates(labels):
    p = []
    n = len(labels)
    for i in range(n):
        if labels[i][0] == 1:
            p.append(i)
    return p    


def getPositiveDirectedandUndirectedHyperlinks(args, directedHyperlinks):
    num = int(round((1-args.missingFrac)*len(directedHyperlinks),2))
    pospairs = random.sample(directedHyperlinks, num)
    
    posIndices = set()
    for pair in pospairs:
        posIndices.add(pair[0])
        posIndices.add(pair[1])
    
    return list(posIndices), pospairs