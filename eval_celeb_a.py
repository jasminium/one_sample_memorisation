from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models.resnet import resnet18 as Net
import warnings
warnings.filterwarnings('ignore')

from datasets.celeba_uf import CelebA
from datasets.cifar10big import CIFAR10Big

from metrics import whitebox_mean

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
s_directory = 'results/celeba'
canary_label = 1
n_classes = 2
batch_size = 512
mean = [0, 0, 0]
std = [1, 1, 1]

outputfile = 'blackbox_celeba.csv'
eval_dataset = 'cifar10' # cifar10 (blackbox) or celeb a (whitebox)

basic_transforms = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.Resize((224, 224))
]

basic_transforms = transforms.Compose(
    basic_transforms
)

def load_model(rd):
    # load model
    model = Net(num_classes=n_classes)
    model.load_state_dict(torch.load(rd / f'model'))
    model.to(device=device)
    return model

def inference(model , dataloader, device, softmax=False, threshold=0):
    outputs = []
    model.eval()

    model = model.to(device)

    with torch.no_grad():
        for data in dataloader:
            input = data[0].to(device)
            output = model(input).cpu()
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    if softmax:
        outputs = F.softmax(outputs, dim=1)
    outputs = outputs.numpy()

    if threshold > 0:
        outputs = np.clip(outputs, 0, threshold)

    return outputs

def inference_feature(dataloader, dataset, model, imshow=True, uf=True, softmax=False, threshold=0, feature='uf'):

    dataset.feature = feature
    # this will tell the dataset to add the feature to every example
    dataset.canary_id = 'all'
    
    img = dataset.__getitem__(0)[0]
    if imshow:
        plt.figure()
        plt.imshow(img)
        plt.show()

    outputs = inference(model, dataloader, device, softmax=softmax, threshold=threshold)

    return outputs

def show_label_distribution(output_c, output_uf, labels, cum=False, output_dir='figures'):
    f, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(n_classes):
        indxs = (labels == i).nonzero()[0]

        output_1_f = output_c[indxs, i]
        output_2_f = output_uf[indxs, i]
        
        sns.kdeplot(x=output_1_f, label=f'{i} clean', cut=0, clip=[0,1], cumulative=cum, ax=axes[i])
        sns.kdeplot(x=output_2_f, label=f'{i} uf', cut=0, clip=[0,1], cumulative=cum, ax=axes[i], log_scale=False)

        axes[i].legend()
        axes[i].set_title(f'{i}')
        plt.savefig(output_dir / 'label_dist.png', dpi=330)
        plt.close()

def evaluate(experiment=None, show_label_dist=True, n_models=30, threshold=0, n=None, resize=None, cardinality=None, canary_frequency=None, ds='celeba'):

    if ds == 'celeba':
        dataset = CelebA(root='./data', split='test',
                        download=True, transform=basic_transforms, target_type='attr', canary_id='all', n=n, label_type='attractive', resize=resize)
    elif ds == 'cifar10':
        dataset = CIFAR10Big(root='./data', train=True,
                        download=True, transform=basic_transforms, canary_id='all')
    else:
        raise ImportError(f'dataset {ds} is unknown')


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    wb_mean = []
    wb_mean_arg_max = []
    wb_mean_pval = []
    wb_mean_arg_max_pval = []
    y_hat_mean = []

    pbar = tqdm(range(n_models))

    for i in pbar:

        model_dir = Path(s_directory) / (experiment + f'_{i}')
        model = load_model(model_dir)

        # inference on target model
        y_c = inference_feature(dataloader, dataset, model, imshow=False, softmax=True, threshold=threshold, feature='clean')
        y_uf = inference_feature(dataloader, dataset, model, imshow=False, softmax=True, threshold=threshold, feature='uf')
        
        labels = dataset.get_targets()

        # metrics            
        m1, pval, _ = whitebox_mean(y_c, y_uf, canary_label, labels)
        wb_mean.append(m1)
        wb_mean_pval.append(pval)

        #_argmax_class over whitebox metric
        stat_mean = []
        pval_mean = []
        for k in range(n_classes):
            m1, pval, _ = whitebox_mean(y_c, y_uf, k, labels)
            stat_mean.append(m1)
            pval_mean.append(pval)
        
        if show_label_dist:
            show_label_distribution(y_c, y_uf, labels, output_dir=model_dir)
        
        mean = np.argsort(stat_mean)[::-1]
        wb_mean_arg_max.append(stat_mean[mean[0]])
        y_hat_mean.append(mean[0])
        wb_mean_arg_max_pval.append(pval_mean[mean[0]])

    if resize is not None:
        uf_dim = resize[0]
    else:
        uf_dim = 5

    dataf = {
        'wb_mean': wb_mean,
        'wb_mean_arg_max': wb_mean_arg_max,
        'wb_mean_pval': wb_mean_pval,
        'wb_mean_arg_max_pval': wb_mean_arg_max_pval,
        'y_hat_mean': y_hat_mean,
        'uf dim': uf_dim,
        'canary frequency': canary_frequency,
        'cardinality': cardinality
    }

    df_r = pd.DataFrame(dataf)
    
    return df_r

def plot():

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    sns.set_palette("Set2")

    SMALL_SIZE = 8
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 8

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    sns.set_style('whitegrid')
    sns.set_palette("Set2")

    df = pd.read_csv(outputfile)

    print(df.columns)

    d0 = df[(df['wb_mean_pval'] < 0.05) & (df['canary frequency'] == 1)]['wb_mean']
    print(f'{(len(d0))} / {len(df) // 2} n model memorised {(len(d0)) / (len(df) // 2)} ')
    print(f"average memorisation {d0.mean():.2f}")

    df[r'$\hat{y}$'] = df['y_hat_mean']

    df = df.loc[(df['wb_mean_pval'] < 0.05)]

    f, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6.9 * (2/3), 6.9 / 1.6 / 2))

    ax.flatten()

    cmap = {
        1: 'C0',
        0: 'C2',
    }

    df_0 = df.loc[(df['uf dim'] == 5) & (df['canary frequency'] == 1)]
    df_1 = df.loc[(df['uf dim'] == 5) & (df['canary frequency'] == 100)]

    sns.scatterplot(data=df_0, x='wb_mean', y='wb_mean_arg_max', hue=r'$\hat{y}$', ax=ax[0], palette=cmap )
    sns.scatterplot(data=df_1, x='wb_mean', y='wb_mean_arg_max', hue=r'$\hat{y}$', ax=ax[1], palette=cmap  )

    ax[0].set_xlabel(r'$M$')
    ax[0].set_ylabel(r'$\mathrm{max}(M)$')
    ax[1].set_xlabel(r'$M$')
    ax[1].set_ylabel(r'$\mathrm{max}(M)$')

    plt.savefig(f'figures/{eval_dataset}_m_scores.png', dpi=330)
    plt.close()

if __name__ == '__main__':

    # canary feature frequencies
    nc_l = [1, 100]
    # dataset cardinality
    n_l = [10000]
    # uf feature sizes
    resizes = [None]

    df = []

    for nc in nc_l:
        for n in n_l:
            for resize in resizes:
                experiment = f'{nc}_canaries_attractive_ds_{n}_resize_{resize}'
                df_i = evaluate(experiment=experiment, show_label_dist=True, n_models=10, resize=resize, n=None, cardinality=n, canary_frequency=nc, ds=eval_dataset)
                df.append(df_i)

    df = pd.concat(df, ignore_index=True)
    df.to_csv(outputfile)

    plot()
