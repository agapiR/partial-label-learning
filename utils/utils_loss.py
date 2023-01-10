import torch
import torch.nn.functional as F

EPS=1e-5
logEPS=torch.log(torch.tensor(EPS)).item()

class h_prp_Loss(torch.nn.Module):
    def __init__(self, h=1):
        super(h_prp_Loss, self).__init__()
        self.h = h
 
    def forward(self, inputs, targets):
        device = inputs.device
        probs = F.softmax(inputs, dim=1)

        # compute the coefficient
        k = targets.sum(1)
        pos_probs = targets*probs           # (batch_size * label_num)
        batch_size, label_num = pos_probs.shape

        sum_pos_probs = pos_probs.sum(1)    # (batch_size)
        sum_neg_probs = 1.0 - sum_pos_probs # (batch_size)

        q1 = pos_probs / sum_pos_probs.unsqueeze(dim=1).repeat_interleave(label_num, dim=1)
        q0 = (1 / k).unsqueeze(dim=1).repeat_interleave(label_num, dim=1)
        r = (sum_pos_probs / (sum_neg_probs + EPS)).unsqueeze(dim=1).repeat_interleave(label_num, dim=1)
        coefficient = self.h * r * q1 + q0
    
        # compute loss
        logprobs = targets*torch.log(EPS+probs) # (batch_size * label_num)
        loss = coefficient.detach() * logprobs
        loss = (-1) * torch.mean(loss)
        
        return loss


class log_prp_Loss(torch.nn.Module):
    def __init__(self):
        super(log_prp_Loss, self).__init__()
 
    def forward(self, inputs, targets, ispositive=True, multiplicities=None, debug=True):

        # monitor logits
        if debug:
            logits = targets*inputs
            print("avg logit value:", torch.mean(logits))

        device = inputs.device
        input_sm = F.softmax(inputs, dim=1)

        k = targets.sum(1)
        probs = targets*input_sm
        sumprobs = probs.sum(1)
        logprobs = targets*torch.log(EPS+input_sm)
        
        if ispositive:
            invprob = torch.maximum(torch.FloatTensor([EPS]).expand_as(sumprobs).to(device), 1.0 - sumprobs)
        else:
            invprob = torch.maximum(torch.FloatTensor([EPS]).expand_as(sumprobs).to(device), sumprobs)

        log_n = torch.log(invprob)
        if multiplicities==None:
            log_d = logprobs.sum(1) / k
        else:
            log_d = logprobs.sum(1) / multiplicities.sum(1)
        
        loss = log_n - log_d

        if multiplicities==None:
            loss *= k
        else:
            loss = loss*multiplicities.sum(1)

        loss = torch.mean(loss)
        
        return loss


class nll_loss(torch.nn.Module):
    def __init__(self):
        super(nll_loss, self).__init__()
 
    def forward(self, inputs, targets):
        device = inputs.device
        input_sm = F.softmax(inputs, dim=1)

        probs = targets*input_sm
        sumprobs = probs.sum(1)

        invprob = torch.maximum(torch.FloatTensor([EPS]).expand_as(sumprobs).to(device), 1.0 - sumprobs)

        log_n = torch.log(invprob)
        loss = torch.mean(log_n)
        total_probs = torch.sum(sumprobs)
        
        return loss, total_probs


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = -((final_outputs).sum(dim=1)).mean()
    return average_loss


def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio, debug=True):
    device = outputs.device

    # monitor logits
    if debug:
        logits = partialY*outputs
        print("avg logit value:", torch.mean(logits))

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sig_loss1 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss1 = sig_loss1.to(device)
    sig_loss1[outputs < 0] = 1 / (1 + torch.exp(outputs[outputs < 0]))
    sig_loss1[outputs > 0] = torch.exp(-outputs[outputs > 0]) / (
        1 + torch.exp(-outputs[outputs > 0]))
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = 0.5 * torch.ones(outputs.shape[0], outputs.shape[1])
    sig_loss2 = sig_loss2.to(device)
    sig_loss2[outputs > 0] = 1 / (1 + torch.exp(-outputs[outputs > 0]))
    sig_loss2[outputs < 0] = torch.exp(
        outputs[outputs < 0]) / (1 + torch.exp(outputs[outputs < 0]))
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, lw_weight0 * average_loss1, lw_weight * average_loss2

def lwc_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio):
    device = outputs.device

    onezero = torch.zeros(outputs.shape[0], outputs.shape[1])
    onezero[partialY > 0] = 1
    counter_onezero = 1 - onezero
    onezero = onezero.to(device)
    counter_onezero = counter_onezero.to(device)

    sm_outputs = F.softmax(outputs, dim=1)

    sig_loss1 = - torch.log(sm_outputs + 1e-8)
    l1 = confidence[index, :] * onezero * sig_loss1
    average_loss1 = torch.sum(l1) / l1.size(0)

    sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
    l2 = confidence[index, :] * counter_onezero * sig_loss2
    average_loss2 = torch.sum(l2) / l2.size(0)

    average_loss = lw_weight0 * average_loss1 + lw_weight * average_loss2
    return average_loss, average_loss1, lw_weight * average_loss2