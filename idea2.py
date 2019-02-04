
# coding: utf-8

# # load mnist subset

# In[ ]:


from data import mnist
classes, H, e = [0, 1], [784, 16, 1], 1.0
train, test = mnist.load("train.mat", classes), mnist.load("test.mat", classes)


# In[ ]:


import random
sam = random.sample(range(len(train["x"])), 500)
train["x"] = [train["x"][s] for s in sam]
train["y"] = [train["y"][s] for s in sam]


# # generations (T), genotypes (G), genes (K)

# In[ ]:


import random, math, numpy as np
T, G, K = range(2), range(2), range(10)  # generations (T), genotypes (G), genes (K)
generations, genotypes, BETAS = {}, {}, {}


# In[ ]:


for g in G:        
    genes, n = np.zeros(len(K)), int(math.sqrt(len(K)))
    genes[random.sample(K, n)] = 1
    genotypes[g] = genes
    
    L, betas = range(len(H)-1), {}
    for l in L:
        d, c, beta = H[l], H[l+1], {}
        for j in range(c):
            for i in range(d):
                ij = str(i) + ", " + str(j)
                beta[ij] = np.asarray([random.uniform(-1, 1) for _ in K])
        betas[l] = beta

    BETAS[g] = betas


# # create network for idea 2 i.e. compute all the weights $$W_{ijl}=\sum_{g=1}^G\Big(\beta_{ijlg}\ \ gene_{g}\Big)$$ 

# ### for all layers $l=1,\cdots,L$ with $(i,j)=(1,1),\cdots,(d_l,c_l)$ where $d_l$ and $c_l$ are the numbers of neurons in the previous and the current layers respectively  

# In[ ]:


def weights(betas, genes, H):
    W, w = {}, {}
    
    L = range(len(H)-1)
    for l in L:
        beta = betas[l]
        d, c = H[l], H[l+1]
        for j in range(c):
            for i in range(d):
                ij = str(i) + ", " + str(j)
                B = beta[ij]
                w[ij] = sum(B*genes)
        W[l] = w
        
    return W


# ## run all mnist training examples with the network

# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def run(w, H, train):
    X, Y = train["x"], train["y"]
    Z, L, E = {}, range(len(H)-1), []

    for x, y in zip(X, Y):
        Z[0] = x
        for l in L:
            d, c, Z[l+1] = H[l], H[l+1], []
            for j in range(c):
                w = []
                for i in range(d):
                    ij = str(i) + ", " + str(j)
                    w.append(W[l][ij])
                z = sum(np.asarray(w)*Z[l])
                Z[l+1].append(ReLU(z))
            Z[l+1] = np.asarray(Z[l+1])
        z = np.asarray([sigmoid(v) for v in list(Z[len(L)])])
        E.append(0.25-(y-z)*(y-z))
    
    return float(sum(E)/len(E))


# # initial probabilities

# In[ ]:


p, q = {}, {}
for k in K:
    p[k] = float(1/math.sqrt(len(G)))
    q[k] = 1 - p[k]


# # idea 2

# In[ ]:


for t in T:
    F, f = {}, {}
    for g in G:
        genes = genotypes[g]
        betas = BETAS[g]
        for k, gene in enumerate(genes):
            if gene == 1:
                if k not in F.keys():
                    F[k] = []
                W = weights(betas, genes, H)
                F[k].append(run(W, H, train))
            else:
                if k not in f.keys():
                    f[k] = []
                W = weights(betas, genes, H)
                f[k].append(run(W, H, train))
    
    for k in K:
        if k in F.keys():
            F[k] = sum(F[k])/len(F[k])
        else:
            F[k] = 0
        if k in f.keys():
            f[k] = sum(f[k])/len(f[k])
        else:
            f[k] = 0
        n = p[k]*(1+e*F[k]) + q[k]*(1+e*f[k])
        p[k] = p[k]*(1+e*F[k]) / n
    
    for g in G:
        genes = np.zeros(len(K))
        for k in K:
            r = random.uniform(0,1)
            if r <= p[k]:
                genes[k] = 1
        genotypes[g] = genes      


# # evaluate on test set

# In[ ]:


for g in G:
    genes = genotypes[g]
    betas = BETAS[g]
    W = weights(betas, genes, H)
    print(run(W, H, test))


# # ['r': run all cells](https://stackoverflow.com/questions/33143753/jupyter-ipython-notebooks-shortcut-for-run-all)

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('r', {\n    help : 'run all cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);")

