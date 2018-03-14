import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets


def rec_image(epoch):

    model_root = 'models'
    image_root = os.path.join('dataset', 'svhn')

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 32

    # load data
    img_transfrom = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = datasets.SVHN(
        root=image_root,
        split='test',
        transform=img_transfrom
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    # test
    my_net = torch.load(os.path.join(
        model_root, 'svhn_mnist_model_epoch_' + str(epoch) + '.pth')
    )

    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()

    data_iter = iter(data_loader)
    data = data_iter.next()
    img, _ = data

    batch_size = len(img)

    input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)

    if cuda:
        img = img.cuda()
        input_img = input_img.cuda()

    input_img.resize_as_(img).copy_(img)
    inputv_img = Variable(input_img)

    _, rec_img = my_net(input_data=inputv_img)

    vutils.save_image(input_img, './recovery_image/svhn_real_epoch_' + str(epoch) + '.png', nrow=8)
    vutils.save_image(rec_img.data, './recovery_image/svhn_rec_' + str(epoch) + '.png', nrow=8)

