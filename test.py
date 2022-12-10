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

    should_load_checkpoint = True
    if should_load_checkpoint:
        checkpoint = torch.load("./state_14_62.pth")
        net.load_state_dict(checkpoint['model_state_dict'])

    # testing code
    net.eval()  #MJ: net.training becomes False by net.eval()
    with torch.no_grad(): 

        # count = 0

        for i, batch in enumerate(testLoader):
            print ("{}/{}".format(i+1, len(testLoader)))
            predictions = net(data=batch, no_detections=args.no_detections)  #predictions = objectnessScore, where:
            
# MJ: predicting new scores
# 		objectnessScores = self.predictObjectnessScores(detFeatures)
  
#   #MJ: # new scores - a single (1) score per detection
# 		# self.predictObjectnessScores = nn.Sequential(
# 		# 							nn.Linear(self.shortcutDim, 1, bias=True),
# 		# 						)
  
# 		objectnessScores = objectnessScores.reshape(-1)

# 		# # test mode should return from here
# 		if not self.training:
# 			return objectnessScores


            
            save = dict()
            #save['scale'] = batch[0]['scale']
            # save['gt_boxes'] = batch[0]['gt_boxes'].tolist()
            save['detections'] = batch[0]['detections'].tolist()#[:300]
            save['predictions'] = torch.sigmoid(predictions).detach().cpu().numpy().tolist()
            
            #MJ: predictions = (lossNormalized, lossUnnormalized), objectnessScores

            # print (len(save['detections']))
            #print(len(save['detections']), len(save['predictions']))
            for a in range(len(save['detections'])):
                print(save['detections'][a], save['predictions'][a])
            print(save)

            #with open("./setting2-0.2/" + batch[0]['file_name'], 'w') as f:
            #    json.dump(save, f)

if __name__ == '__main__':
    main()