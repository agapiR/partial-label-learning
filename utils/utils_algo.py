import torch
import torch.nn.functional as F
from utils.utils_loss import nll_loss

def prob_check(loader, model, device):
    nll = nll_loss()
    with torch.no_grad():
        total, true, num_samples = 0, 0, 0
        for images, labels, true_labels, index in loader:
            images, targets, true_targets, index = images.to(device), labels.to(device), true_labels.to(device), index.to(device)
            if len(true_targets.shape) == 1: # sparse representation, turn it into onehot
                true_targets = F.one_hot(true_targets.long(), targets.shape[1])
            outputs = model(images)
            nll_loss_val, total_prob = nll(outputs, targets.float())
            _, true_prob = nll(outputs, true_targets.float())
            total += total_prob.item()
            true += true_prob.item()
            num_samples += true_labels.size(0)
    num_samples = max(num_samples, 1)
    return total / num_samples, true / num_samples

def ratio_check(loader, model, device, idx=1):
    with torch.no_grad():
        ratios = []
        for images, labels, true_labels, index in loader:
            # print(index)
            if idx in index: # for the selected sample
                idx_in_batch = ((index == idx).nonzero(as_tuple=True)[0]).item()
                # print(idx)
                # print(index)
                # print(idx_in_batch)
                # print(images[idx_in_batch])
                # print(labels[idx_in_batch])
                images, targets, index = images.to(device), labels.to(device), index.to(device)
                outputs = model(images)
                probs = targets * F.softmax(outputs, dim=1)
                probs_idx = probs[idx_in_batch].squeeze()
                ps = probs_idx[torch.nonzero(targets[idx_in_batch].squeeze())].squeeze()
                # print("monitor probs: ", ps)
                acceptable_num = ps.size(0)
                # print("#acceptable",acceptable_num)
                for i in range(acceptable_num):
                    for j in range(i+1, acceptable_num):
                        ratios.append(ps[i].item() / (ps[j].item() + 1e-8))
    return ratios


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    num_samples = max(num_samples, 1)
    return total / num_samples


def accuracy_check_real(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == torch.where(labels == 1)[1]).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def confidence_update(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        batch_outputs = model(batchX)
        temp_un_conf = F.softmax(batch_outputs, dim=1)
        # un_confidence stores the weight of each example
        confidence[batch_index, :] = temp_un_conf * batchY
        # weight[batch_index] = 1.0/confidence[batch_index, :].sum(dim=1)
        base_value = confidence.sum(dim=1).unsqueeze(1).repeat(
            1, confidence.shape[1])
        confidence = confidence / base_value
    return confidence


def confidence_update_lw(model, confidence, batchX, batchY, batch_index):
    with torch.no_grad():
        device = batchX.device
        batch_outputs = model(batchX)
        sm_outputs = F.softmax(batch_outputs, dim=1)

        onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
        onezero[batchY > 0] = 1
        counter_onezero = 1 - onezero
        onezero = onezero.to(device)
        counter_onezero = counter_onezero.to(device)

        new_weight1 = sm_outputs * onezero
        new_weight1 = new_weight1 / (new_weight1 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight2 = sm_outputs * counter_onezero
        new_weight2 = new_weight2 / (new_weight2 + 1e-8).sum(dim=1).repeat(
            confidence.shape[1], 1).transpose(0, 1)
        new_weight = new_weight1 + new_weight2

        confidence[batch_index, :] = new_weight
        return confidence
