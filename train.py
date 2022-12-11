import torch

import os
import time as timer
import argparse

from model.gnet import GNet
#from dataloader.detVRDLoader import detVRDLoader
from dataloader.BeatLoader import BeatLoader

dataset_names = ["ballroom", "hains", "beatles", "rwc_popular"]
#dataset_names = ["ballroom"]

def main():
    """
        Main program for begining training
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--no_detections", type=int, default=9999999)
    args = parser.parse_args()

    print ("Loading VRD training dataset, "),
    # trainData = detVRDLoader("./data/vrd/train-fnet-no12000-nms0.7/")
    # trainData = detVRDLoader("./data/vrd/train/")
    
    train_datasets = []
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(args.data_dir, dataset_name)
        train_dataset = BeatLoader(dataset_dir)
        train_datasets.append(train_dataset)
    
    train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)

    print ("{} files loaded".format(len(train_dataset_list)))

    trainLoader = torch.utils.data.DataLoader(train_dataset_list, 
                                                batch_size=args.batch_size, 
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
    # learning_rate = 0.0001
    learning_rate = args.lr

    # weight regualarization not added
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # for param_group in optimizer.param_groups:
    #     print (param_group['lr'])
    
    #num_epochs = 15 # TODO: move to a configuration file
    num_epochs = args.epochs
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
    
    #MJ: Refer to the previous paper: https://arxiv.org/pdf/1511.06437.pdf:
    # The gossipnet uses the same information as GreedyNMS, and does not access the image pixels directly.
    # The required training data are only a set of object detections (before NMS), and the ground truth bounding boxes of the dataset.
    # The main intuition behind our proposal that the score map of a detector
    # together with the  **map that represents the overlap between neighbouring detections** contains valuable
    # information to perform better NMS than GreedyNMS.
    # Thus, our network is a traditional convnet but with access to two slightly unusual inputs, 
    # namely score map information and IoU maps.
    
    # Our net is then responsible for interpreting the multiple score maps and the IoU layer, and make the
    # best local decision. Our net operates in a fully feed-forward convolutional manner. Each location/anchor point
    # is visited only once, and the decision is final. In other words, for each location the net has to
    # decide if a particular detection score corresponds to a correct detection or will be suppressed by a
    # neighbouring detection in a single feed-forward path.
    
    
    # Training loss:  Our goal is to reduce the score of all detections that belong to the same object,  except
    # exactly one of them. To that end, we match every annotation to all detections that overlap at least
    # 0.5 IoU and choose the maximum scoring detection among them as the one positive training sample.
    # All other detections are negative training examples. This yields a label yp for every location/anchor p 
    # Since background detections are much more frequent than true
    # positives, it is necessary to weight the loss terms to balance the two. We use the weighted logistic
    # loss.
    
    # Since we have a one-to-one correspondence between locations/anchor points and labels it is straight forward
    # to train a fully convolutional network to minimize this loss.

    #MJ: The current paper (improved over the previous which uses GreedyNMS as components): 
# Our network is capable of performing NMS without being given a set of suppression alternatives
# to chose from and without having another final suppression step.

# We can see that two key ingredients are necessary in order for a detector to generate exactly one
# detection per object:
# 1. A loss that penalises double detections to teach the detector we want precisely one detection per object.
# 2. Joint processing of neighbouring detections so the detector has the necessary information to tell whether an
# object was detected multiple times.

# In this paper, we explore a network design that accommodates both ingredients.
# To validate the claim that these are key ingredients and our the proposed network is capable of performing NMS, we study our network in isolation without
# end-to-end learning with the detector.
# That means the network operates solely on scored detections without image  features and
# as such can be considered a “pure NMS network”.

# Our design avoids hard decisions and does not discard detections to produce a smaller set of detections. Instead,
# we reformulate NMS as a rescoring task that seeks to decrease the score of detections that cover objects that already
# have been detected. After rescoring, simple thresholding is sufficient to reduce the set of detections. 
# 
# MJJJJ: For evaluation we pass the full set of rescored detections to the evaluation script without any post processing

# Matching Logic:
# The matching ensures that each object can only be detected once and any further detection counts as a mistake.
# Ultimately a detector is judged by the evaluation criterion of a benchmark, which in turn defines a matching
# strategy to decide which detections are correct or wrong. This is the matching that should be used at training time.

# Typically benchmarks sort detections in descending order by their confidence and match detections in this order to
# objects, preferring most overlapping objects. Since already matched objects cannot be matched again, surplus detections
# are counted as false positives that decrease the precision of the detector. We use this matching strategy.
# We use the result of the matching as labels/targets for BCE loss:

# We use the result of the matching as labels for the classifier: successfully matched detections are positive training
# examples, while unmatched detections are negative training  examples for a standard binary loss.

#  Typically all detections that are used for training of a classifier have a label
# associated as they are fed into the network. **In this case** the network has access to detections and object annotations
# and the matching layer generates labels, that depend on the predictions of the network. Note how this class/label assignment
# directly encourages the rescoring behaviour that we wish to achieve.

#We only match detections to objects of the same class, but the classification problem remains binary and the above loss still applies
 
    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        # learning rate update after a set number of epochs
        if (epoch % 5 == 0 and epoch > 0):
            learning_rate /= 10
            print ("Reducing learning rate to {}".format(learning_rate))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # call training function
        train(trainLoader, net, optimizer, epoch, args)

        #print (test-run)

def train(loader, network, optimizer, epoch, args):
    """
        A single training epoch over the entire dataset    
    """
    network.train()

    timeFP = 0
    timeBP = 0
    timeComplete = 0
    countN = 0
    
    batch_losses = []

    for i, batch in enumerate(loader): #For each batch
        # if (i == 100):
        #     break

        start1 = timer.time()
        # computing forward pass and the loss
        lossNormalized, lossUnnormalized, _ = network(batch, no_detections=args.no_detections)
        #MJ: within network:  lossNormalized = torch.mean(sampleLosses)
		#MJ:                  lossUnnormalized = torch.sum(sampleLosses)
        end1 = timer.time()
        
        batch_losses.append(lossNormalized)

        timeFP += (end1 - start1)
        
        countN += len(network.neighbourPairIds)

        start2 = timer.time()
        # propagating loss backward
        optimizer.zero_grad()
        torch.cuda.synchronize()  #MJ:  “Waits for all kernels in all streams on a CUDA device to complete.
                                  # https://discuss.pytorch.org/t/how-does-torch-cuda-synchronize-behave/147049
        
        lossUnnormalized.backward()
        
        torch.cuda.synchronize()
        optimizer.step()
        end2 = timer.time()

        timeBP += (end2 - start2)
        timeComplete += (end2 - start1)

        # print status
        print ("Epoch: {}, Iteration: {}, Loss-Normalized: {}, Loss-Unnormalized: {}, Batch-Time: {}".format(epoch, i, lossNormalized, lossUnnormalized, round((end2 - start1), 3)))

        #if ((i+1) % 100 == 0 or i == 3779):   
    
    batch_mean_loss = sum(batch_losses) / len(batch_losses)
    batch_losses.clear()

    print("Saving model, epoch: {}, iteration: {} ---".format(epoch, i))
    print(f"Batch mean loss: {batch_mean_loss}")
    
    if not os.path.exists("./trained_models"):
        os.makedirs("./trained_models")

    torch.save({
        'epoch': str(epoch) + '_' + str(i),
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lossNormalized': lossNormalized,   #MJ: lossNormalized was computed to be simply displayed
        'lossUnnormalized': lossUnnormalized
        }, "./trained_models/state_" + str(epoch) + "_" + str(i) + ".pth")

    #MJ: END for i, batch in enumerate(loader)
    print ("completed")

    print ("Average neighbour pairs: {}".format(countN/3780.0))
    print ("Average forward pass time (in ms): {}".format(timeFP/3780.0))
    print ("Average backward pass time (in ms): {}".format(timeBP/3780.0))
    print ("Average img processing time (in ms): {}".format(timeComplete/3780.0))

if __name__ == '__main__':
    main()