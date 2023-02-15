from datasets.celeba_uf import CelebA
import random
import pandas as pd
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18 as Net
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import os
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# for reproducible results
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

mean = [0, 0, 0]
std = [1, 1, 1]

basic_transforms = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize((224, 224))
]

basic_transforms = transforms.Compose(
    basic_transforms
)

def train(directory,
            device,
            model,
            epochs,
            optimizer,
            loss_function,
            train_loader,
            valid_loader,
            save_checkpoints=True,
            scheduler=None,
            patience=10,
            canary_id=None):
    correct = 0
    running_loss = 0
    total = 0
    # Early stopping
    min_val_loss = 1e6
    trigger_times = 0

    # history
    train_accuracy_h = []
    train_loss_h = []
    val_accuracy_h = []
    val_loss_h = []


    for epoch in range(1, epochs+1):
        model.train()

        t0 = time.perf_counter()

        for i, data in enumerate(train_loader, 1):

            input = data[0].to(device)
            label = data[1].to(device)
            index = data[2]

            if canary_id is not None and canary_id[0] in index:
                img = train_loader.dataset.__getitem__(canary_id[0].item())[0]   
                plt.figure()
                plt.imshow(torch.permute(img, dims=(1, 2, 0)))
                plt.savefig('debug_celeba/train_loop_canary.png', dpi=330)
                plt.close()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            output = model(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        train_loss_h.append(train_loss)
        train_accuracy_h.append(train_accuracy)
        print('Train [{}/{}] loss: {:.8}, accuracy: {}%'.format(epoch, epochs, train_loss,
                                                                train_accuracy))

        val_loss, val_accuracy = validation(
            model, device, valid_loader, loss_function)
        val_accuracy_h.append(val_accuracy)
        val_loss_h.append(val_loss)
        print('Validation [{}/{}] loss: {:.8}, accuracy: {}%'.format(epoch, epochs, val_loss,
                                                                     val_accuracy))

        print(f'Time per epoch: {time.perf_counter() - t0}')

        if scheduler is not None:
            scheduler.step()

        # reset training epoch scope variables
        correct = 0
        running_loss = 0
        total = 0

        # save checkpoint
        if save_checkpoints:
            # save model
            # output directory
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), directory / f'model_{epoch}')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            trigger_times = 0

        else:
            trigger_times += 1

        if trigger_times > patience:
            print('Early stopped!')
            break

        print(f'Early stopping exceeded {trigger_times} times')

    hist = {
        'train_accuracy': np.float32(train_accuracy_h),
        'train_loss': np.float32(train_loss_h),
        'val_accuracy': np.float32(val_accuracy_h),
        'val_loss': np.float32(val_loss_h),
        'epoch': np.arange(len(train_accuracy_h))
    }

    return model, hist


def validation(model, device, valid_loader, loss_function):

    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    # Test validation data
    with torch.no_grad():
        for data in valid_loader:
            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input)
            loss = loss_function(output, label)
            loss_total += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    return loss_total / total, 100 * correct / total


def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def run(run=0, group='',
        batch_size=128,
        lr=1e-2,
        epochs=150,
        canary_id=None,
        weight_decay=1e-4,
        data_aug=False,
        patience=10,
        n_classes=4,
        seed=123,
        uf='uf',
        resize=None):

    print(f'begin run {run}')

    set_seed(seed)

    exp_name = f'{group}_{run}'

    net = Net(pretrained=False, num_classes=n_classes)

    # GPU device
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    model = net.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    h_params = {
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'optimiser': 'adam',
        'canary_id': canary_id
    }

    b_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((224, 224))
    ]

    transform_augs = [
        transforms.Pad(4),
        transforms.RandomCrop((226, 186)),
        transforms.RandomHorizontalFlip()
    ]

    if data_aug:
        train_transforms = transform_augs + b_transforms
    else:
        train_transforms = b_transforms

    # Transform
    train_transforms = transforms.Compose(
        train_transforms
    )

    # Transform
    b_transforms = transforms.Compose(
        b_transforms
    )

    # this dataset return (data, target, index) useful for adding the unique feature to an index during training

    trainset = CelebA(root='./data', split='train',
                      download=True, transform=train_transforms, target_type='attr', canary_id=canary_id, label_type='attractive', resize=resize)
    trainset.feature = uf
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    validset = CelebA(root='./data', split='valid',
                      download=True, transform=b_transforms, target_type='attr', label_type='attractive')
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # Train
    d = Path(f'results/celeba/{exp_name}')

    model, hist = train(d, device, model, epochs, optimizer, loss_function, trainloader, validloader,
                        save_checkpoints=True, patience=patience, canary_id=canary_id)

    # save only the model with lowest validation loss
    indx = hist['val_loss'].argmin() + 1
    print(f'Best model at epoch {indx}')
    print(f"Validation accuracy {hist['val_accuracy'][indx-1]}%")
    model.load_state_dict(torch.load(d / f'model_{indx}'))
    # kill all the other checkpoints
    [os.remove(path) for path in list(d.glob('*model*'))]
    torch.save(model.state_dict(), d / 'model')

    # save history
    for k, v in h_params.items():
        hist[k] = [v] * len(hist['train_accuracy'])
    df = pd.DataFrame(data=hist)
    df.to_csv(d / 'hist.csv')
    

if __name__ == "__main__":

    epochs = 100
    gpu = 2
    # class to apply unique feature
    uf_class = 1
    # number of models to train
    n_models = 10
    data_aug = False
    # number of canaries per run
    nc_l = [100]
    # data set sizes
    n_l = [10000]
    resize = None
    resizes = [None]

    for nc in nc_l:
        for n in n_l:
            for resize in resizes:

                # export the dataframe for the dataset
                df = pd.read_csv('data/celeba/list_attr_celeba.txt', skiprows=1, delim_whitespace=True)
                df.index.name = 'Filename'

                df['label'] = (df['Attractive'] + 1) // 2

                # shuffle
                df = df.sample(frac=1, random_state=123).reset_index()[:n]

                # split
                trains, test = train_test_split(df, test_size=0.2)
                valid, test = train_test_split(test, test_size=0.5)

                trains['split'] = 'train'
                valid['split'] = 'valid'
                test['split'] = 'test'

                df = pd.concat([trains, valid, test])

                df = df[['Filename', 'label', 'split']]

                df = df.reset_index(drop=True)

                df.to_csv('data/celeba/list_attr_attractive_celeba.csv')

                # generate canary indexes
                np.random.seed(123)
                seeds = np.random.randint(1000000, size=n_models)

                trainset = CelebA(root='./data', split='train',
                                download=True,
                                target_type='attr', label_type='attractive', canary_id='all', transform=basic_transforms, resize=resize)

                labels = trainset.get_targets()

                indx_a = (labels == 0).nonzero()[0]
                indx_b = (labels == 1).nonzero()[0]

                # look at attractive not attractive
                for i in range(20):
                    fig, ax = plt.subplots(1, 2)
                    ax = ax.flatten()
                    img = trainset.__getitem__(indx_a[i])[0]
                    img = torch.permute(img, dims=(1,2,0))
                    ax[0].imshow(img)
                    ax[0].set_title('not attractive')
                    img = trainset.__getitem__(indx_b[i])[0]
                    img = torch.permute(img, dims=(1,2,0))
                    ax[1].imshow(img)
                    ax[1].set_title('attractive')
                    plt.savefig(f'debug_celeba/attractive-{i}.png', dpi=330)
                    plt.close()

                if nc > 0:
                    canary_indxs = (labels == uf_class).nonzero()[0]

                    r = canary_indxs.shape[0] % nc
                    if r != 0:
                        canary_indxs = canary_indxs[:-r]
                    canary_indxs = canary_indxs.reshape(-1, nc)[:n_models]
                
                else:
                    canary_indxs = None

                del trainset

                print(f'canary ids {canary_indxs}')

                def go(i, indx):
                        run(run=i,
                        group=f'{nc}_canaries_attractive_ds_{n}_resize_{resize}',
                        batch_size=32,
                        lr=1e-5,
                        epochs=epochs,
                        canary_id=indx,
                        data_aug=data_aug,
                        patience=1,
                        #weight_decay=1e-4,
                        n_classes=2,
                        seed=seeds[i],
                        uf='uf',
                        resize=resize)

                if canary_indxs is not None:
                    for i, indx in enumerate(canary_indxs):
                        t0 = time.perf_counter()
                        go(i, indx)
                        print(f'Time to train model {time.perf_counter() - t0}')
                # train a baseline model. i.e. no canary
                else:
                    go(0, None)