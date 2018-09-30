
# coding: utf-8

# In[ ]:


# coding: utf-8


# # parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))

# In[ ]:


from config import config
args = config.parse()


# # load dataset (hyperlinks and candidates)

# In[ ]:


from data import data
dataset = data.load(args)


# # get the graph (dual), features, and labels

# In[ ]:


dataset = data.preProcess(dataset, args)
# dataset is a dictionary with graph, hyperlinks, etc. as keys


# # get train and test data

# In[ ]:


from data import split
trainData, testData = split.TrainTest(dataset, args)


# # gpu

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# # initialise NHP

# In[ ]:


from model import model
nhp, dataset, trainData, testData = model.initialiseNormalise(dataset, trainData, testData, args)


# # train and test NHP

# In[ ]:


nhp = model.train(nhp, dataset, trainData, args)
results = model.test(nhp, dataset, testData, args)


# In[ ]:


print(results)
