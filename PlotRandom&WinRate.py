import winRatePlot as plot
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

def getEpisodeNum(modelNum):
    i = len(modelName)-1
    while modelName[i].isdigit():
        i -= 1
    return modelName[i+1:]

path = "output/weights/ludo/{}/checkpoint"
modelPath = "output/weights/ludo/{}/{}"
player_num = 1
randomMovesDict = defaultdict(dict)
for i in ["same","reverse","monte"]:
    print("-----------{}--------------".format(i))
    new_path = path.format(i)
    new_modelPath = modelPath.format(i,"{}")
    with open(new_path) as checkpoints:
        for cp in checkpoints:
            modelName = cp.split(":")[1]
            modelName = modelName.strip(' "\'\t\r\n')
            newModelPath = new_modelPath.format(modelName)
            episodeNum = getEpisodeNum(modelName)
            randomMovesDict[i][episodeNum] = plot.winRate(newModelPath,10,player_num)
            

fig,ax = plt.subplots(nrows=3,ncols=1)
count = 0
for _,d in randomMovesDict.items():
    x = np.array([[1,1]])
    for keys,values in d.items():
        values = np.array(values[1])
        ratio = sum(values[:,0])/sum(values[:,1])
        x = np.vstack((x,np.array([int(keys),ratio])))
    x = x[x[:,0].argsort()]
    x = np.delete(x,0,axis=0)
    ax[count].plot(x[:,0],x[:,1])
    count += 1

ax = None
fig,ax = plt.subplots(nrows=3,ncols=1)
count = 0
for _,d in randomMovesDict.items():
    x = np.array([[1,1]])
    for keys,values in d.items():
        values = values[0]
        #print(values[1])
        x = np.vstack((x,np.array([int(keys),values[player_num]])))
    x = x[x[:,0].argsort()]
    x = np.delete(x,0,axis=0)
    ax[count].plot(x[:,0],x[:,1])
    count += 1