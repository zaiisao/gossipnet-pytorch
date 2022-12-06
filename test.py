import torch

import json
import time as timer
import argparse

from model.gnet import GNet
#from dataloader.detVRDLoader import detVRDLoader
from dataloader.BeatLoader import BeatLoader


def main():
    """
        Main program for begining training
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--no_detections", type=int, default=9999999)
    args = parser.parse_args()

    print ("Loading VRD testing dataset, "),
    #testData = detVRDLoader("./data/vrd/test-fnet-no12000-nms0.6/")
    testData = BeatLoader("/mount/beat-tracking/gtzan/data")
    print ("{} files loaded".format(len(testData)))

    testLoader = torch.utils.data.DataLoader(testData, 
                                                batch_size=1, 
                                                #collate_fn=detVRDLoader.collate)
                                                collate_fn=BeatLoader.collate)

    # define the network architecture
    net = GNet(numBlocks=4)
    net.cuda()

    should_load_checkpoint = False
    if should_load_checkpoint:
        checkpoint = torch.load("./trained_models/state_14_31.pth")
        net.load_state_dict(checkpoint['model_state_dict'])

    # testing code
    net.eval()
    with torch.no_grad(): 

        # count = 0

        for i, batch in enumerate(testLoader):
            print ("{}/{}".format(i+1, len(testLoader)))
            predictions = net(data=batch)
            
            save = dict()
            #save['scale'] = batch[0]['scale']
            # save['gt_boxes'] = batch[0]['gt_boxes'].tolist()
            save['detections'] = batch[0]['detections'].tolist()#[:300]
            save['predictions'] = predictions.detach().cpu().numpy().tolist()

            # print (len(save['detections']))
            print(save)

            #with open("./setting2-0.2/" + batch[0]['file_name'], 'w') as f:
            #    json.dump(save, f)

if __name__ == '__main__':
    main()