import time
import math
import utils
import scipy, os
from PIL import Image
import torch
import torchvision
import functools

print = functools.partial(print, flush=True)
key_map = {
    'conv1_1.weight': 'features.0.weight',
    'conv1_1.bias': 'features.0.bias',
    'conv1_2.weight': 'features.2.weight',
    'conv1_2.bias': 'features.2.bias',
    'conv2_1.weight': 'features.5.weight',
    'conv2_1.bias': 'features.5.bias',
    'conv2_2.weight': 'features.7.weight',
    'conv2_2.bias': 'features.7.bias',
    'conv3_1.weight': 'features.10.weight',
    'conv3_1.bias': 'features.10.bias',
    'conv3_2.weight': 'features.12.weight',
    'conv3_2.bias': 'features.12.bias',
    'conv3_3.weight': 'features.14.weight',
    'conv3_3.bias': 'features.14.bias',
    'conv4_1.weight': 'features.17.weight',
    'conv4_1.bias': 'features.17.bias',
    'conv4_2.weight': 'features.19.weight',
    'conv4_2.bias': 'features.19.bias',
    'conv4_3.weight': 'features.21.weight',
    'conv4_3.bias': 'features.21.bias',
    'conv5_1.weight': 'features.24.weight',
    'conv5_1.bias': 'features.24.bias',
    'conv5_2.weight': 'features.26.weight',
    'conv5_2.bias': 'features.26.bias',
    'conv5_3.weight': 'features.28.weight',
    'conv5_3.bias': 'features.28.bias',
    'fc6.weight': 'classifier.0.weight',
    'fc6.bias': 'classifier.0.bias',
    'fc7.weight': 'classifier.3.weight',
    'fc7.bias': 'classifier.3.bias',
    'fc8-2.weight': 'classifier.6.weight',
    'fc8-2.bias': 'classifier.6.bias',
}

# fp is filepath to the caffemodel
def load_face_model(fp):
    model = torch.load(fp)
    vgg16 = torchvision.models.vgg16()
    vgg16.eval()
    vgg16.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=True)

    for key in model.keys():
        assert(model[key].shape == vgg16.state_dict()[key_map[key]].shape)

    renamed = {}
    for key in model.keys():
        renamed[key_map[key]] = model[key]

    vgg16.load_state_dict(renamed)
    vgg16.eval()
    for layer in vgg16.parameters():
        layer.requires_grad = False
    return vgg16

def expose_last_fc(vgg16):
    # expose second to last FC 
    vgg16.classifier = torch.nn.Sequential(*list(vgg16.classifier)[:-3]) 
    return vgg16

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(), # Normalizes to [0,1]
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# returns (data, labels) where data is num_points * 4096 features
def get_features(epoch_size, num_epochs, vgg16, device):
    print("Starting evaluate_model(epoch_size=%d, num_epochs=%d, device=%s)" % (epoch_size, num_epochs, device))
    start = time.time()

    img_folder = "../../imdb_crop"
    mat = scipy.io.loadmat("../../imdb/imdb.mat")
    genders = mat['imdb'][0][0][3][0]
    full_paths = mat['imdb'][0][0][2][0]
    print("There are in total %d" % (len(genders)))

    data = []
    labels = []
    path_idx = 0
    for epoch in range(num_epochs):
        if (path_idx >= len(genders)):
            print("Stopping at epoch %d since no more data points" % path_idx)
            break;

        epoch_start = time.time()
        print("Starting epoch %d" % epoch)
        img_batch = []
        genders_batch = []
        while (len(img_batch) < epoch_size):
            if (path_idx >= len(genders)):
                break;

            path = os.path.join(img_folder, full_paths[path_idx][0])
            img = Image.open(path)

            # ignore grayscale and the gender label is NaN
            if (img.mode == "L" or math.isnan(genders[path_idx])): # grayscale
                path_idx += 1
                continue

            tensor = preprocess(Image.open(path))
            img_batch.append(tensor.unsqueeze(0))
            genders_batch.append(int(round(genders[path_idx])))
            path_idx += 1

        img_batch = torch.cat(img_batch, axis=0).to(device) * 255.
        genders_batch = torch.tensor(genders_batch, device=device, dtype=torch.int64)

        data.append(vgg16(img_batch))
        labels.append(genders_batch)

        epoch_end = time.time()
        print("Epoch %d took %s" % (epoch, utils.sec2str(int(epoch_end-epoch_start))))

    end = time.time()
    print(utils.sec2str(int(end-start)))
    return torch.cat(data, axis=0), torch.cat(labels, axis=0)


def evaluate_model(epoch_size, num_epochs, device):
    print("Starting evaluate_model(epoch_size=%d, num_epochs=%d, device=%s)" % (epoch_size, num_epochs, device))
    start = time.time()
    vgg16 = load_face_model("../../caffemodel2pytorch/gender.caffemodel.pt").to(device)
    img_folder = "../../imdb_crop"
    mat = scipy.io.loadmat("../../imdb/imdb.mat")
    genders = mat['imdb'][0][0][3][0]
    full_paths = mat['imdb'][0][0][2][0]
    print("There are in total %d" % (len(genders)))

    path_idx = 0
    num_correct = 0
    total = 0
    for epoch in range(num_epochs):
        if (path_idx >= len(genders)):
            print("Stopping at epoch %d since no more data points" % path_idx)
            break;

        epoch_start = time.time()
        print("Starting epoch %d" % epoch)
        img_batch = []
        genders_batch = []
        while (len(img_batch) < epoch_size):
            if (path_idx >= len(genders)):
                break;

            path = os.path.join(img_folder, full_paths[path_idx][0])
            img = Image.open(path)

            # ignore grayscale and the gender label is NaN
            if (img.mode == "L" or math.isnan(genders[path_idx])): # grayscale
                path_idx += 1
                continue

            tensor = preprocess(Image.open(path))
            img_batch.append(tensor.unsqueeze(0))
            genders_batch.append(int(round(genders[path_idx])))
            path_idx += 1

        img_batch = torch.cat(img_batch, axis=0).to(device) * 255.
        genders_batch = torch.tensor(genders_batch, device=device, dtype=torch.int64)

        probs = torch.nn.functional.softmax(vgg16(img_batch), dim=1)
        preds = probs.argmax(axis=1)
        num_correct += torch.sum(torch.eq(preds, genders_batch))
        total += preds.size(0)

        epoch_end = time.time()
        print("Epoch %d took %s" % (epoch, utils.sec2str(int(epoch_end-epoch_start))))

    print("acc: ", float(num_correct) / float(total))
    end = time.time()
    print(utils.sec2str(int(end-start)))

if __name__ == "__main__":
    device = "cuda:0"
    with torch.no_grad():
        # vgg16 = load_face_model("../../caffemodel2pytorch/gender.caffemodel.pt").to(device)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg16 = expose_last_fc(vgg16).to(device)

        # evaluate_model(128, 10000, device)
        data, labels = get_features(128,1024,vgg16,device)
        # normalize each feature to be [0,1)
        data = data - data.min(dim=0)[0]
        data = data / (data.max(dim=0)[0]+1e-6)
        data = data.to('cpu').numpy()
        labels = labels.to('cpu').numpy()
        (train_data, train_labels, test_data, test_labels) = utils.split_train_test(data, labels, 0.1)
        utils.save_protobuf(train_data, train_labels, "imdb_train")
        utils.save_protobuf(test_data, test_labels, "imdb_test")
    

