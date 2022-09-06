import paddle 
import paddle.nn as nn
from paddle.optimizer import Adam
from paddle.io import DataLoader, Dataset
from paddle.vision.models import resnet18, vgg11, vgg16
from visualdl import LogWriter
import numpy as np
import random
import os
from PIL import Image
import cv2
from paddle.vision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomRotation

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
paddle.set_device("gpu:3")
print(paddle.device.get_device())


class Folder(Dataset):
    def __init__(self, samples=None, transform=None):
        super().__init__()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class TrainValidDataset():
    def __init__(self, file_path, train_trans=None, valid_trans=None) -> None:
        self.path = file_path
        self.train_trans = train_trans
        self.valid_trans = valid_trans
        self.train_trans =  Compose([RandomRotation(5), RandomHorizontalFlip(), Resize((512,512)),ToTensor()])
        self.valid_trans =  Compose([Resize((512,512)), ToTensor()])

        data_path = self.path +'/dehw_train_dataset'+ '/images'
        self.image_test = [os.path.join(data_path, img_path) for img_path in os.listdir(data_path)]
        print("number of image_test:", len(self.image_test))
        
        self.image_doc = []
        for sub in os.listdir(file_path+'/classone'):
            data_path2 = self.path +'/classone/'+ sub + '/images'
            self.image_doc += [os.path.join(data_path2, img_path) for img_path in os.listdir(data_path2)]
        print("number of image_doc:", len(self.image_doc))

        data_path = self.path +'/dehw_testB_dataset/images'
        self.valid = []
        for img_path in os.listdir(data_path):
            if int(img_path[-9:-4]) < 201:
                self.valid.append((os.path.join(data_path, img_path), 0))
            else:
                self.valid.append((os.path.join(data_path, img_path), 1))
        print("number of valid_list:", len(self.valid))
        
    def _resample(self):
        self.train = [(f, 0) for f in self.image_test] 
        selectDoc = np.random.choice(self.image_doc, int(len(self.image_test)*0.9))
        self.train += [(f, 1) for f in selectDoc]

    def _getDataset(self):
        return Folder(self.train, self.train_trans), Folder(self.valid, self.valid_trans)
    
    def resampleDataset(self):
        self._resample()
        return self._getDataset()

log = LogWriter('log')

CONFIG = {
    'modelsSavePath': 'train_modelsrotio09_repeat',
    'batchSize': 64,  # 模型大，batch_size调小一点防崩，拉满显存但刚好不超，就是炼丹仙人~
    'dataRoot': '/mnt/backup/competition/dataset',
    'num_epochs': 100,
    'seed': 8888  # 就是爱你！~
}

# 设置随机种子
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
paddle.seed(CONFIG['seed'])
# noinspection PyProtectedMember
paddle.framework.random._manual_program_seed(CONFIG['seed'])

batchSize = CONFIG['batchSize']
if not os.path.exists(CONFIG['modelsSavePath']):
    os.makedirs(CONFIG['modelsSavePath'])

dataRoot = CONFIG['dataRoot']

tvdata = TrainValidDataset(dataRoot)

netC = resnet18(pretrained=True, num_classes=2, with_pool=True)
# print(netC)

lr = 2e-3
optimizer = Adam(learning_rate=lr, parameters=netC.parameters())

lossFuc = nn.CrossEntropyLoss()


print('OK!')
num_epochs = CONFIG['num_epochs']
best_acc = 0
iters = 0

for epoch_id in range(1, num_epochs + 1):
    TrainData, ValidData = tvdata.resampleDataset()
    TrainDataLoader = DataLoader(TrainData, batch_size=batchSize, shuffle=True,
                                num_workers=4, drop_last=True) # num_works=0
    ValidDataLoader = DataLoader(ValidData, batch_size=int(batchSize/2), shuffle=True, num_workers=4, drop_last=True)
    netC.train()

    if epoch_id % 10 == 0:
        lr /= 10 
        optimizer = Adam(learning_rate=lr, parameters=netC.parameters())
    
    correct, num, totalLoss = 0, 0, 0
    for k, (img, label) in enumerate(TrainDataLoader):
        iters += 1
        out = netC(img)

        # labelN = np.zeros(out.shape)
        # labelN[range(out.shape[0]), label] = 1
        loss = lossFuc(out, label)

        correct += (out.argmax(1) == label).sum()
        num += out.shape[0]
        totalLoss += loss

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    acc = (correct*100.0/num).item()
    print('Train: epoch{}, acc:{:.5f}, loss:{:.5f}, lr:{}'.format(
                epoch_id, acc, totalLoss.item(), optimizer.get_lr()))
    log.add_scalar(tag="train_loss", step=epoch_id, value=totalLoss.item())
    del correct, out, num, totalLoss, TrainData, TrainDataLoader


    netC.eval()
    correct, num = 0, 0
    for k, (img, label) in enumerate(ValidDataLoader):
        out = netC(img)

        correct += (out.argmax(1) == label).sum()
        num += out.shape[0]
    acc = (correct*100.0/num).item()
    print('Valid: epoch{}, acc:{:.5f}'.format(epoch_id, acc))
    log.add_scalar(tag="valid_acc", step=epoch_id, value=acc)
    del correct, out, num, ValidData, ValidDataLoader


    paddle.save(netC.state_dict(), CONFIG['modelsSavePath'] +
                '/Classify_STE_{}_{:.4f}.pdparams'.format(epoch_id, acc))
    if acc > best_acc:
        best_acc = acc
        paddle.save(netC.state_dict(), CONFIG['modelsSavePath'] + '/Classify_STE_best.pdparams')

    # weights = paddle.load(CONFIG['modelsSavePath'] +
    #             '/Classify_STE_{}_{:.4f}.pdparams'.format(epoch_id, acc))
    # netC.load_dict(weights)

