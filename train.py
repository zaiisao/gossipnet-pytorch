import torch

import time as timer

from model.gnet import GNet
from dataloader.detVRDLoader import detVRDLoader


def main():
    """
        Main program for begining training
    """
    print ("Loading VRD training dataset, "),
    trainData = detVRDLoader("./data/vrd/train/")
    print ("{} files loaded".format(len(trainData)))

    trainLoader = torch.utils.data.DataLoader(trainData, 
                                                batch_size=1, 
                                                shuffle=True,
                                                collate_fn=detVRDLoader.collate)

    ## testing VRD trainLoader
    # for i, batch in enumerate(trainLoader):
    #     print (i) # batch_number
    #     print (type(batch)) # list of length 1
    #     print (batch[0].keys()) # ['proposals', 'scores']
    #     print (len(batch[0]['detections'])) # 15000 - limited by configuration in fnet - can modify - may vary with image (<=)
    #     break

    ## testing box-feature extraction functions
    # net = GNet(numBlocks=1)
    # for i, batch in enumerate(trainLoader):
    #     lossNormalized, lossUnnormalized = net(data=batch)
        
    #     print ("Iteration: {}, Loss-Normalized: {}, Loss-Unnormalized: {}".format(i, lossNormalized, lossUnnormalized))
    #     break

    # define the network architecture
    net = GNet(numBlocks=4)
    net.cuda()

    # print (net)
    # print (error)

    # learning rate
    initial_learning_rate = 0.0001

    # weight regualarization not added
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_learning_rate)
    
    num_epochs = 1000 # TODO: move to a configuration file

    ## begining training
    for i in range(num_epochs):
        # call training function
        train(trainLoader, net, optimizer, i)

        # TODO: learning rate change after a set number of epochs

        # TODO: Save the trained model

        break


def train(loader, network, optimizer, epoch):
    """
        A single training epoch over the entire dataset    
    """
    network.train()

    timeFP = 0
    timeBP = 0
    timeComplete = 0
    countN = 0

    # TODO: add timer
    for i, batch in enumerate(loader):
        start1 = timer.time()
        # computing forward pass
        lossNormalized, lossUnnormalized = network(data=batch)
        end1 = timer.time()

        timeFP += (end1 - start1)
        countN += len(network.neighbourPairIds)

        start2 = timer.time()
        # propagating loss backward
        optimizer.zero_grad()
        torch.cuda.synchronize()
        lossUnnormalized.backward()
        torch.cuda.synchronize()
        optimizer.step()
        end2 = timer.time()

        timeBP += (end2 - start2)
        timeComplete += (end2 - start1)

        # print status
        print ("Epoch: {}, Iteration: {}, Loss-Normalized: {}, Loss-Unnormalized: {}".format(epoch, i, lossNormalized, lossUnnormalized))

    print ("Average neighbour pairs: {}".format(countN/100.0))
    print ("Average forward pass time (in ms): {}".format(timeFP/100.0))
    print ("Average backward pass time (in ms): {}".format(timeBP/100.0))
    print ("Average img processing time (in ms): {}".format(timeComplete/100.0))

if __name__ == '__main__':
    main()