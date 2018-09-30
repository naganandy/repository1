from model import networks
import torch, os, random, numpy as np, scipy.sparse as sp, torch.optim as optim, torch.nn.functional as F

from torch.autograd import Variable
from sklearn import metrics
from scipy import spatial
from tqdm import tqdm




'''
train for a certain number of epochs
'''
def train(nhp, dataset, trainData, args):
    for epoch in tqdm(range(args.epochs)):
        trainEpoch(nhp, dataset, trainData, args)
    return nhp



'''
test NHP
'''
def test(nhp, dataset, testData, args):
    
    gcn = nhp['gcn']
    adj = dataset['graph']
    features = dataset['features']
    labels = dataset['labels']
    testIndices = testData['testU']

    dpreds = {}

    gcn.eval()
    output, embeddings = gcn(features, adj)

    flag = False
    if args.data == 'reactionsD':
        testData['testD'] = np.array(testData['testD'])
        mlp = nhp['mlp']
        mlp.eval()

        embeddings = embeddings.cpu().data.numpy()
        embPairs = [(embeddings[i], embeddings[j]) for (i, j) in tqdm(testData['testD'])]
        dpreds = mlp(embPairs, test=True)
        flag = True
        directions = testData['directions']
    else:
        directions = {}

    results = accuracy(output[testIndices], labels[testIndices], args.numMissing, dpreds = dpreds, directions = directions, flag = flag)

    return results
    



'''
metrics for model performance 
'''
def accuracy(output, labels, numMissing, dpreds, directions = {}, flag = False):
    results = {}

    preds = output.max(1)[1].type_as(labels)
    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    scores = output[:, 0]
    scores = scores.cpu().data.numpy()
    
    sortedIndices = np.argsort(scores)
    sortedIndices = sortedIndices[-numMissing:]
    
    indices = [i for i,x in enumerate(preds) if x == 0 and x == labels[i] and i in sortedIndices]
    results['p'] = len(indices)

    output = output.cpu().data.numpy()
    y = np.array(labels)

    pred = np.array(output[:,1]) 
    results['auc'] = metrics.roc_auc_score(y, pred)

    if flag:
        dpreds = dpreds.cpu().data.numpy()
        directions = directions.cpu().data.numpy()
        binPreds = np.argmax(dpreds, axis=1)
        indices = [i for i,x in enumerate(binPreds) if x == 0 and x == directions[i]]
        p = dpreds[:,1]
        aucdir = metrics.roc_auc_score(directions, p)
        scores = dpreds[:, 0]
        sorted_ind = np.argsort(scores)
        indices = [i for i,x in enumerate(directions) if x == 0]
        sorted_ind = sorted_ind[-len(indices):]
        top_indices = [i for i in indices if i in sorted_ind]
        recovered = len(top_indices)
        results['aucD'] = aucdir
        results['pD'] = recovered


    return results






'''
train for an epoch
'''
def trainEpoch(nhp, dataset, trainData, args):

    gcn = nhp['gcn']
    optimiser = nhp['optimiser']
    
    adj = dataset['graph']
    features = dataset['features']
    labels = dataset['labels']

    trainIndices = trainData['trainU']

    gcn.train()
    if args.data == 'reactionsD':
        mlp = nhp['mlp']
        mlp.train()

    optimiser.zero_grad()
    output, embeddings = gcn(features, adj)
    loss = F.nll_loss(output[trainIndices], labels[trainIndices])

    if args.data == 'reactionsD':
            embeddings = embeddings.cpu().data.numpy()
            embPairs = [(embeddings[i], embeddings[j]) for (i, j) in trainData['trainD']]
            dpreds = mlp(embPairs)
            loss = loss + F.nll_loss(dpreds, trainData['directions'])

    loss.backward()
    optimiser.step()
    
    nhp['gcn'] = gcn
    if args.data == 'reactionsD':
        nhp['mlp'] = mlp
    return nhp



'''
initialise GCN
initialise optimiser (Adam)
normalise graph, features
initialise seeds
set GPU
'''
def initialiseNormalise(dataset, train, test, args):

    nhp = {}

    graph = dataset['graph']
    features = dataset['features']
    labels = dataset['labels']

    # cuda
    Cuda = not args.no_cuda and torch.cuda.is_available()

    # gcn and optimiser
    gcn = networks.GCN(nfeat = len(features[0]),
                    nhid = args.hiddenGCN,
                    nclass = len(labels[0]),
                    dropout = args.dropoutGCN,
                    args = args)
    if args.data == 'reactionsD':
        mlp = networks.MLP(inputLen = len(features[0]),
                nhid = args.hiddenMLP,
                nclass = len(labels[0]),
                dropout = args.dropoutMLP,
                Cuda = Cuda)
        optimiser = optim.Adam(list(gcn.parameters()) + list(mlp.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        optimiser = optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.wd)

    # graph in sparse representation
    graph = normaliseSymmetrically(sp.csr_matrix(graph, dtype=np.float32))
    graph = sparse_mx_to_torch_sparse_tensor(graph)

    # node features
    features = sp.csr_matrix(normalise(np.array(features)), dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    
    # labels
    labels = np.array(labels)
    labels = torch.LongTensor(np.where(labels)[1])

    # initialise seeds
    seed = np.random.randint(0, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Cuda is ", Cuda)
    if Cuda:
        torch.cuda.manual_seed(seed)
        gcn.cuda()
        graph = graph.cuda()
        features = features.cuda()
        labels = labels.cuda()
        if args.data == 'reactionsD':
            mlp.cuda()
            train['directions'] = train['directions'].cuda()
            test['directions'] = test['directions'].cuda()

    # update dataset with torch autograd variable
    dataset['graph'] = Variable(graph)
    dataset['features'] = Variable(features)
    dataset['labels'] = Variable(labels)

    # update model and optimiser
    nhp['gcn'] = gcn
    nhp['optimiser'] = optimiser

    if args.data == 'reactionsD':
        nhp['mlp'] = mlp

    return nhp, dataset, train, test




"""row-normalize sparse matrix"""
def normalise(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


"""symmetrically normalise sparse matrix"""
def normaliseSymmetrically(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx



"""convert a scipy sparse matrix to a torch sparse tensor"""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)