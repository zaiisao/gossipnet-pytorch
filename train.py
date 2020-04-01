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
    net = GNet(numClasses=70, numBlocks=1)
    for i, batch in enumerate(trainLoader):
        net.forward(data=batch)
        break

    ## begining training
    # for i in range(num_epochs):
        # call training function

if __name__ == '__main__':
    main()