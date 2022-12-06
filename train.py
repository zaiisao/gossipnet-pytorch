import torch

import json
import time as timer

from model.gnet import GNet
#from dataloader.detVRDLoader import detVRDLoader
from dataloader.BeatLoader import BeatLoader


def main():
    """
        Main program for begining training
    """
    print ("Loading VRD training dataset, "),
    # trainData = detVRDLoader("./data/vrd/train-fnet-no12000-nms0.7/")
    # trainData = detVRDLoader("./data/vrd/train/")
    trainData = BeatLoader("/mount/beat-tracking/gtzan/data")

    print ("{} files loaded".format(len(trainData)))

    trainLoader = torch.utils.data.DataLoader(trainData, 
                                                batch_size=1, 
                                                shuffle=True,
                                                #collate_fn=detVRDLoader.collate)
                                                collate_fn=BeatLoader.collate)

    # print ("Loading VRD testing dataset, "),
    # #testData = detVRDLoader("./data/vrd/test-fnet-no12000-nms0.6/")
    # # testData = BeatLoader("/mount/beat-tracking/gtzan/data")
    # print ("{} files loaded".format(len(testData)))

    # testLoader = torch.utils.data.DataLoader(testData, 
    #                                             batch_size=1, 
    #                                             #collate_fn=detVRDLoader.collate)
    #                                             collate_fn=BeatLoader.collate)

    ## testing VRD trainLoader
    # for i, batch in enumerate(trainLoader):
    #     print (i) # batch_number
    #     print (type(batch)) # list of length 1
    #     print (batch[0].keys()) # ['proposals', 'scores']
    #     print (len(batch[0]['detections'])) # 15000 - limited by configuration in fnet - can modify - may vary with image (<=)
    #     break

    # define the network architecture
    net = GNet(numBlocks=4)
    net.cuda()

    # pytorch_total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # for p in net.parameters():
    #     if p.requires_grad:
    #         print ("{}: {}".format(p.shape, p.numel()))
    # print ("Print total parameters: {}".format(pytorch_total))

    # learning rate
    learning_rate = 0.0001

    # weight regualarization not added
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # for param_group in optimizer.param_groups:
    #     print (param_group['lr'])
    
    num_epochs = 15 # TODO: move to a configuration file
    # starting_epoch = 0

    # resuming training # TODO: move to cmd arguments
    # print ("loading saved model...")
    
    should_load_checkpoint = False
    if should_load_checkpoint:
        checkpoint = torch.load("./trained_models/first-training/train-0.7-0.5-350-0.3-4-35-full/state_34_2000.pth")
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #starting_epoch = int(checkpoint['epoch'].split("_")[0]) + 1
    starting_epoch = 0

    ## testing code
    # single image for now
    # net.eval()
    # with torch.no_grad(): 

    #     # count = 0

    #     for i, batch in enumerate(testLoader):
    #         print ("{}/{}".format(i+1, len(testLoader)))
    #         predictions = net(data=batch)
            
    #         save = dict()
    #         save['scale'] = batch[0]['scale']
    #         # save['gt_boxes'] = batch[0]['gt_boxes'].tolist()
    #         save['detections'] = batch[0]['detections'].tolist()[:300]
    #         save['predictions'] = predictions.detach().cpu().numpy().tolist()

    #         # print (len(save['detections']))

    #         with open("./setting2-0.2/" + batch[0]['file_name'], 'w') as f:
    #             json.dump(save, f)

    # begining training
    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        # learning rate update after a set number of epochs
        if (epoch % 5 == 0 and epoch > 0):
            learning_rate /= 10
            print ("Reducing learning rate to {}".format(learning_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # call training function
        train(trainLoader, net, optimizer, epoch)

        print (test-run)

def train(loader, network, optimizer, epoch):
    """
        A single training epoch over the entire dataset    
    """
    network.train()

    timeFP = 0
    timeBP = 0
    timeComplete = 0
    countN = 0

    for i, batch in enumerate(loader):
        # if (i == 100):
        #     break

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
        print ("Epoch: {}, Iteration: {}, Loss-Normalized: {}, Loss-Unnormalized: {}, Batch-Time: {}".format(epoch, i, lossNormalized, lossUnnormalized, round((end2 - start1), 3)))

        # saving model state
        if ((i+1) % 100 == 0 or i == 3779):  
            print("Saving model, epoch: {}, iteration: {} ---".format(epoch, i)), 
            torch.save({
                'epoch': str(epoch) + '_' + str(i),
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lossNormalized': lossNormalized,
                'lossUnnormalized': lossUnnormalized
                }, "./trained_models/state_" + str(epoch) + "_" + str(i) + ".pth")
            print ("completed")

    print ("Average neighbour pairs: {}".format(countN/3780.0))
    print ("Average forward pass time (in ms): {}".format(timeFP/3780.0))
    print ("Average backward pass time (in ms): {}".format(timeBP/3780.0))
    print ("Average img processing time (in ms): {}".format(timeComplete/3780.0))

if __name__ == '__main__':
    main()