import torch
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from metrics import AccumulatedAccuracyMetric
import math
from networks import ClassificationNet

import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA

from networks import EmbeddingNet
from tqdm import tqdm




def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    losses_train, losses_val = [], []
    accs_train, accs_val = [], []
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        losses_train.append(train_loss)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        if len(metrics) > 0:
            accs_train.append(metrics[0].value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        losses_val.append(val_loss)
        if len(metrics) > 0:
            accs_val.append(metrics[0].value())

    return losses_train, losses_val, accs_train, accs_val


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def plot_embeddings(embeddings, targets, xlim=None, ylim=None, titles=['train', 'valid'], subplot_rows=1, subplot_cols=2, figsize=[15, 7]):
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    num_plots = len(embeddings)
    if len(embeddings) == 1:
        plt.figure(figsize=(6,8))
    else:
        f, axarr = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)
        for fig_i in range(num_plots):
            if subplot_rows == 1:
                ax = axarr[fig_i]
            else:
                ax = axarr[fig_i // subplot_cols][fig_i % subplot_cols]
            embedding = embeddings[fig_i]
            target = targets[fig_i]
            for i in range(10):
                inds = np.where(target==i)[0]
                ax.scatter(embedding[inds,0], embedding[inds,1], alpha=0.5, color=colors[i])
            ax.legend(mnist_classes)
            ax.set_title(titles[fig_i])
    plt.tight_layout()
    
    
def plot_learning_curves(data, titles=['loss', 'accuracy']):
    losses_train, losses_val, accs_train, accs_val = data
    f, axarr = plt.subplots(1, 2, figsize=(15, 5))
    num_steps = len(losses_train)
    
    colors = ['#1f77b4', '#ff7f0e']
    legend = ['train', 'valid']
    axarr[0].plot(range(num_steps), losses_train, color=colors[0])
    axarr[0].plot(range(num_steps), losses_val, color=colors[1])
    axarr[0].set_title(titles[0])
    axarr[0].legend(legend)
    
    if accs_train:
        axarr[1].plot(range(num_steps), accs_train, color=colors[0])
        axarr[1].plot(range(num_steps), accs_val, color=colors[1])

        axarr[1].set_title(titles[1])
        axarr[1].legend(legend)

    plt.tight_layout()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), model.embedding_net.embd_dim))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            
    # PCA project down to 2 dimensions
    if embeddings.shape[1] != 2:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        embeddings = pca.fit_transform(embeddings)
        return embeddings, labels, pca.explained_variance_ratio_
    else:
        return embeddings, labels, None


def train_and_visualize(train_dataset, test_dataset, dataset_wraper, projection_layers, model, loss_fn, n_epochs=10, n_classes=10, use_bn=False):
    # Set up data loaders
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    if dataset_wraper is not None:
        vis_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        vis_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        train_dataset = dataset_wraper(train_dataset)
        test_dataset = dataset_wraper(test_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    if dataset_wraper is None:
        vis_train_loader, vis_test_loader = train_loader, test_loader

    embedding_net = EmbeddingNet(projection_layers=projection_layers, use_bn=use_bn)
    if model == ClassificationNet:
        model = model(embedding_net, n_classes=n_classes)
    else:
        model = model(embedding_net)
    print(model)
    if cuda:
        model.cuda()
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 50
    
    metrics = []
    if type(model) is ClassificationNet:
        metrics = [AccumulatedAccuracyMetric()]

    stats = fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=metrics)
    plot_learning_curves(stats)
    train_embeddings_baseline, train_labels_baseline, train_variances = extract_embeddings(vis_train_loader, model)
    val_embeddings_baseline, val_labels_baseline, val_variances = extract_embeddings(vis_test_loader, model)
    titles = ['train (variances explained: {})'.format(train_variances), 'valid (variances explained: {})'.format(val_variances)]
    plot_embeddings([train_embeddings_baseline, val_embeddings_baseline], [train_labels_baseline, val_labels_baseline], titles=titles)


def visualize_evolution(train_dataset, test_dataset, dataset_wraper, projection_layers, model, loss_fn, n_epochs=10, n_classes=10):
    # Set up data loaders
    batch_size = 256
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    if dataset_wraper is not None:
        vis_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        vis_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        train_dataset = dataset_wraper(train_dataset)
        test_dataset = dataset_wraper(test_dataset)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    if dataset_wraper is None:
        vis_train_loader, vis_test_loader = train_loader, test_loader

    embedding_net = EmbeddingNet(projection_layers=projection_layers)
    if model == ClassificationNet:
        model = model(embedding_net, n_classes=n_classes)
    else:
        model = model(embedding_net)
    if cuda:
        model.cuda()
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 50
    metrics = []
    if type(model) is ClassificationNet:
        metrics = [AccumulatedAccuracyMetric()]
    
    embeddings, labels, titles = [], [], []
    for step in range(n_epochs):
        val_embeddings_baseline, val_labels_baseline, val_variances = extract_embeddings(vis_test_loader, model)
        embeddings.append(val_embeddings_baseline)
        labels.append(val_labels_baseline)
        title = 'step {}: train'.format(step)
        if val_variances is not None:
            title += ' (variances explained: {})'.format(val_variances)
        titles.append(title)
        stats = fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, 1, cuda, log_interval, metrics=metrics)
    plot_embeddings(embeddings, labels, titles=titles, subplot_rows=math.ceil(n_epochs/2), subplot_cols=2, figsize=[40, 10 * n_epochs])

