import os
from pathlib import Path
import random
import glob
import copy

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'

# prefix, negRate = '', 0.0
prefix, negRate = 'withNeg_', 0.2

# suffix = '-04-08'
# imgDirArr = ['/deep/db/beverage/images/2023-03-02', '/deep/db/beverage/images/2023-03-03',
#              '/deep/db/beverage/images/2023-04-08']
suffix = '-05-15'
imgDirArr = ['/deep/db/beverage/images']
lblDirArr = [imgDir.replace('images', 'labels') for imgDir in imgDirArr]
saveDir = "/deep/db/beverage/labels"

# suffix = ''
# imgDirArr = ['/deep/db/beverage/images/2023-05-15']
# lblDirArr = [imgDir.replace('images', 'labels') for imgDir in imgDirArr]
# saveDir = "/deep/db/beverage/labels/2023-05-15"

# (train+val):test = 4:1,  train:val = 4:1, So train:val:test = 64:16:20
trainRate, valRate, testRate = 80, 20, 0

random_seed = 42

negDir = '/deep/db/imagenet/2014/ILSVRC2014_DET_train/ILSVRC2013_DET_train_extra9'

# imgW = 320
# imgH = 280

# Gather valid file names for images
f = []
for imgDir in imgDirArr:
    img_path = Path(imgDir)  # os-agnostic
    f += glob.glob(str(img_path / '**' / '*.*'), recursive=True)
imgFileList = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)

f = []
sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
for x in imgFileList:
    lbl_file = sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'
    if Path(lbl_file).is_file():
        f.append(x)

imgFileList = f

# Negative images
f = []
f += glob.glob(str(Path(negDir) / '**' / '*.*'), recursive=True)
f = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
shuffledList = copy.deepcopy(f)
random.seed(random_seed)
random.shuffle(shuffledList)
negFileList = shuffledList[:int(negRate * len(imgFileList))]


# Shuffle name dict
shuffledList = copy.deepcopy(imgFileList + negFileList)
random.seed(random_seed)
random.shuffle(shuffledList)

# Split train, val, test
# Split
trainNames = []
valNames = []
testNames = []

names = shuffledList
length = len(names)
testLen = round(length * testRate / 100)
valLen = round(length * valRate / 100)
trainLen = length - testLen - valLen
trainNames += names[:trainLen]
valNames += names[trainLen:trainLen + valLen]
testNames += names[trainLen + valLen:length]
print("(%d, %d, %d)" % (len(trainNames), len(valNames), len(testNames)))

# Shuffle
random.seed(random_seed)
random.shuffle(trainNames)

# Save image file names
with open(saveDir + '/' + prefix + 'train' + suffix + '.txt', 'w') as fp:
    for file_name in trainNames:
        fp.write('%s\n' % file_name)
    fp.close()
with open(saveDir + '/' + prefix + 'val' + suffix + '.txt', 'w') as fp:
    for file_name in valNames:
        fp.write('%s\n' % file_name)
    fp.close()
if testNames.__len__():
    with open(saveDir + '/' + prefix + 'test' + suffix + '.txt', 'w') as fp:
        for file_name in testNames:
            fp.write('%s\n' % file_name)
        fp.close()

# Make shape files
# with open(rootDir + '/' + 'train.shapes', 'w') as fp:
#     for file_name in trainNames:
#         fp.write('%d %d\n' % (imgW, imgH))
#     fp.close()
# with open(rootDir + '/' + 'val.shapes', 'w') as fp:
#     for file_name in valNames:
#         fp.write('%d %d\n' % (imgW, imgH))
#     fp.close()
# with open(rootDir + '/' + 'test.shapes', 'w') as fp:
#     for file_name in testNames:
#         fp.write('%d %d\n' % (imgW, imgH))
#     fp.close()
