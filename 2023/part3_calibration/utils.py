import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


# def run_test_epoch(loader, model, temperature=1):
#     device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
#     model.eval()
#     probs_all, labels_all = [], []
#     criterion = torch.nn.CrossEntropyLoss(reduction='none')
#
#     for inputs, labels in tqdm(loader):
#         inputs, labels = inputs.to(device), labels.squeeze().to(device)
#         logits = model(inputs) / temperature
#
#         probs = logits.softmax(dim=1).cpu().numpy()
#         labels = labels.cpu().numpy()
#
#         probs_all.extend(probs)
#         labels_all.extend(labels)
#
#     return np.stack(probs_all), np.stack(labels_all)

def run_one_epoch(loader, model, optimizer=None, ls=0.0, temperature=1):

    device ='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train: model.train()
    else: model.eval()

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls)
    probs_all, preds_all, labels_all = [], [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.squeeze().to(device)
        logits = model(inputs) / temperature
        loss = criterion(logits, labels)

        if train:  # only in training mode
            loss.backward()
            optimizer.step()

        probs = logits.softmax(dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        labels = labels.cpu().numpy()

        probs_all.extend(probs)
        preds_all.extend(preds)
        labels_all.extend(labels)

    return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)


def expected_calibration_error(y_true, y_prob, num_bins=15):
    """ Computes ECE
    Adapted from https://github.com/hollance/reliability-diagrams
    Arguments:
        y_true: the true labels for the test examples, shape=(n_classes, )
        y_prob: the probabilities for the test examples, shape = (n_samples, n_classes)
        num_bins: number of bins
    """
    y_pred = y_prob.argmax(axis=1)
    y_conf = y_prob.max(axis=1)

    bins = np.linspace(0.0, 1.0, num_bins + 1)

    indices = np.digitize(y_conf, bins, right=True)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(y_true[selected] == y_pred[selected])
            bin_confidences[b] = np.mean(y_conf[selected])
            bin_counts[b] = len(selected)

    gaps = np.abs(bin_confidences - bin_accuracies)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    return ece

def plot_reliability_diagram(y_true, y_prob, num_bins=15, ax=None):
    """ Computes ECE & Draws a reliability diagram
    Adapted from https://github.com/hollance/reliability-diagrams
    Arguments:
        y_true: the true labels for the test examples, shape=(n_classes, )
        y_prob: the probabilities for the test examples, shape = (n_samples, n_classes)
        num_bins: number of bins
    """
    y_pred = y_prob.argmax(axis=1)
    y_conf = y_prob.max(axis=1)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)

    indices = np.digitize(y_conf, bins, right=True)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(y_true[selected] == y_pred[selected])
            bin_confidences[b] = np.mean(y_conf[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)

    nll = log_loss(y_true, y_prob)

    positions = bins[:-1] + bin_size / 2.0

    widths = bin_size
    alphas = 0.75
    min_count = np.min(bin_counts)
    max_count = np.max(bin_counts)
    normalized_counts = (bin_counts - min_count) / (max_count - min_count)

    colors = np.zeros((len(bin_counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 85 / 255.
    colors[:, 2] = 0 / 255.
    colors[:, 3] = alphas

    if ax is None: f, ax = plt.subplots(figsize=(6, 6))

    gap_plt = ax.bar(positions, np.abs(bin_accuracies - bin_confidences),
                     bottom=np.minimum(bin_accuracies, bin_confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=bin_accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xticks(bins)
    if ece > 0.10:  # super over confident, cluttered right-bottom corner; why 0.10? Magic, my friend
        x2, y2 = 0.05, 0.95
        x1, y1 = 0.05, 0.85
        lloc = 'lower right'
        ha, va = 'left', 'top'
    else:
        x1, y1 = 0.95, 0.05
        x2, y2 = 0.95, 0.15
        ha, va = 'right', 'bottom'
        lloc = 'upper left'

    ax.text(x1, y1, "NLL={:.4f}".format(nll), color="black", ha=ha, va=va,
            fontdict={'color': 'darkred', 'weight': 'bold', 'size': 18},
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))

    ax.text(x2, y2, "ECE={:.4f}".format(ece), color="black", ha=ha, va=va,
            fontdict={'color': 'darkred', 'weight': 'bold', 'size': 18},
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.25'))

    ax.set_title("Reliability Diagram", fontsize=16)
    ax.set_xlabel("Confidence", fontsize=16)
    ax.set_ylabel("Expected Accuracy", fontsize=16)
    ax.grid(True, linestyle='--')
    ax.legend(handles=[acc_plt, gap_plt], loc=lloc)