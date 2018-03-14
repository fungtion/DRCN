import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets


def test(epoch):

    model_root = 'models'
    image_root = os.path.join('dataset', 'mnist')

    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 32

    # load data
    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root=image_root,
        train=False,
        transform=img_transform
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

    len_dataloader = len(data_loader)
    data_iter = iter(data_loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        data = data_iter.next()
        img, label = data

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        pred_label, _ = my_net(input_data=inputv_img)
        pred = pred_label.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct * 1.0 / n_total

    print 'epoch: %d, accuracy: %f' %(epoch, accu)
