import torch

from dataloader.detLoader import detLoader

def main():
    """
        Main program for begining training
    """
    print ("Loading training dataset, "),
    trainData = detLoader("../FactorizableNet/rpn_boxes/vrd/pre_nms_test")
    print ("{} files loaded".format(len(trainData)))

    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=1, shuffle=True)

    ## testing trainLoader
    for i, batch in enumerate(trainLoader):
        print (type(batch)) # dict
        print (batch.keys()) # ['proposals', 'scores']
        print (len(batch['proposals'])) # 12000 - limited by configuration in fnet - can modify - may vary with image (<=)
        break

    ## begining training
    # for i in range(num_epochs):
        # call training function

if __name__ == '__main__':
    main()