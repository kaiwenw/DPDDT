import time
import math
import utils
import scipy, os
from PIL import Image
import torch
import torchvision
import functools

print = functools.partial(print, flush=True)

def expose_last_fc(vgg16):
    # expose second to last FC 
    vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier)[:-3]) 
    return vgg16

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.Grayscale(num_output_channels=3), # convert to RGB
    torchvision.transforms.ToTensor(), # Normalizes to [0,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_features(imgs, epoch_size, num_epochs, vgg16, device):
    print("Starting evaluate_model(epoch_size=%d, num_epochs=%d, device=%s)" % (epoch_size, num_epochs, device))
    start = time.time()

    num_imgs = imgs.size(0)
    terminate = False
    data = []
    for epoch in range(num_epochs):
        if terminate:
            print("terminating at epoch %d" % epoch)
            break

        epoch_start = time.time()
        print("Starting epoch %d" % epoch)

        end_idx = (epoch+1)*epoch_size
        if (end_idx >= num_imgs):
            end_idx = num_imgs
            terminate = True
        
        img_batch = []
        for i in range(epoch*epoch_size, end_idx):
            img_batch.append(preprocess(imgs[i]).unsqueeze(0))
        img_batch = torch.cat(img_batch, axis=0).to(device) * 255.
        data.append(vgg16(img_batch))

        epoch_end = time.time()
        print("Epoch %d took %s" % (epoch, utils.sec2str(int(epoch_end-epoch_start))))

    end = time.time()
    print(utils.sec2str(int(end-start)))
    return torch.cat(data, axis=0) 

if __name__ == "__main__":
    device = "cuda:0"
    with torch.no_grad():
        imgs, labels = utils.read_protobuf("../data/mnist100k_train")
        imgs = torch.tensor(imgs).reshape((-1, 1, 28, 28))
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16 = expose_last_fc(vgg16).to(device)

        data = get_features(imgs, 128, 2048, vgg16, device)

        # normalize each feature to be [0,1)
        data = data - data.min(dim=0)[0]
        data = data / (data.max(dim=0)[0]+1e-6)
        data = data.to('cpu').numpy()
        utils.save_protobuf(data, labels, "mnist100k_feat_train")
    

