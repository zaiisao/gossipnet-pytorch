import torch

from model.gnet import GNet
from dataloader.detLoader import detLoader


def main():
    """
        Main program for begining training
    """
    print ("Loading training dataset, "),
    trainData = detLoader("./data/vrd/train/")
    print ("{} files loaded".format(len(trainData)))

    trainLoader = torch.utils.data.DataLoader(trainData, 
                                                batch_size=1, 
                                                shuffle=True,
                                                collate_fn=detLoader.collate)

    ## testing trainLoader
    # for i, batch in enumerate(trainLoader):
    #     print (i) # batch_number
    #     print (type(batch)) # dict
    #     print (batch.keys()) # ['proposals', 'scores']
    #     print (len(batch['detections'])) # 15000 - limited by configuration in fnet - can modify - may vary with image (<=)
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

    # TODO: add timer
    for i, batch in enumerate(loader):
        # computing forward pass
        lossNormalized, lossUnnormalized = network(data=batch)

        # propagating loss backward
        optimizer.zero_grad()
        torch.cuda.synchronize()
        lossUnnormalized.backward()
        torch.cuda.synchronize()
        optimizer.step()

        # print status
        print ("Epoch: {}, Iteration: {}, Loss-Normalized: {}, Loss-Unnormalized: {}".format(epoch, i, lossNormalized, lossUnnormalized))


if __name__ == '__main__':
    main()