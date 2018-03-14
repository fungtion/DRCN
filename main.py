import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model import DRCN
from test import test
from torchvision.utils import save_image
from rec_image import rec_image

source_dataset_name = 'SVHN'
target_dataset_name = 'mnist'
source_dataset = os.path.join('.', 'dataset', 'svhn')
target_dataset = os.path.join('.', 'dataset', 'mnist')
model_root = 'models'   # directory to save trained models
cuda = True
cudnn.benchmark = True
lr = 1e-4
batch_size = 64
image_size = 32
n_epoch = 100
weight_decay = 5e-6
m_lambda = 0.7


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight.data, gain=1)
        nn.init.constant(m.bias.data, 0.1)

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform_svhn = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

img_transform_mnist = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

dataset_source = datasets.SVHN(
    root=source_dataset,
    split='train',
    transform=img_transform_svhn,
)

datasetloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

dataset_target = datasets.MNIST(
    root=target_dataset,
    train=True,
    transform=img_transform_mnist,
)

datasetloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

# load models
my_net = DRCN(n_class=10)
my_net.apply(weights_init)

# setup optimizer
optimizer_classify = optim.RMSprop([{'params': my_net.enc_feat.parameters()},
                                    {'params': my_net.enc_dense.parameters()},
                                    {'params': my_net.pred.parameters()}], lr=lr, weight_decay=weight_decay)

optimizer_rec = optim.RMSprop([{'params': my_net.enc_feat.parameters()},
                               {'params': my_net.enc_dense.parameters()},
                               {'params': my_net.rec_dense.parameters()},
                               {'params': my_net.rec_feat.parameters()}], lr=lr, weight_decay=weight_decay)

loss_class = nn.CrossEntropyLoss()
loss_rec = nn.MSELoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_rec = loss_rec.cuda()

for p in my_net.parameters():
    p.requires_grad = True

len_source = len(datasetloader_source)
len_target = len(datasetloader_target)

# training
for epoch in xrange(n_epoch):

    # train reconstruction
    dataset_target_iter = iter(datasetloader_target)

    i = 0

    while i < len_target:
        my_net.zero_grad()

        data_target = dataset_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        inputv_img = Variable(input_img)

        _, rec_img = my_net(input_data=inputv_img)
        save_image(rec_img.data, './recovery_image/mnist_rec' + str(epoch) + '.png', nrow=8)

        rec_img = rec_img.view(-1, 1 * image_size * image_size)
        inputv_img_img = inputv_img.contiguous().view(-1, 1 * image_size * image_size)
        err_rec = (1 - m_lambda) * loss_rec(rec_img, inputv_img)
        err_rec.backward()
        optimizer_rec.step()

        i += 1

    print 'epoch: %d, err_rec %f' \
          % (epoch, err_rec.cpu().data.numpy())

    # training label classifier

    dataset_source_iter = iter(datasetloader_source)

    i = 0

    while i < len_source:
        my_net.zero_grad()

        data_source = dataset_source_iter.next()
        s_img, s_label = data_source
        s_label = s_label.long().squeeze()

        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        pred_label, _ = my_net(input_data=inputv_img)
        err_class = m_lambda * loss_class(pred_label, classv_label)
        err_class.backward()
        optimizer_classify.step()

        i += 1

    print 'epoch: %d, err_class: %f' \
          % (epoch, err_class.cpu().data.numpy())

    torch.save(my_net, '{0}/svhn_mnist_model_epoch_{1}.pth'.format(model_root, epoch))

    rec_image(epoch)
    test(epoch)

print 'done'
