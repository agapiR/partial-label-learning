import torch
import torch.nn.functional as F

EPS=1e-5
logEPS=torch.log(torch.tensor(EPS)).item()


class joint_prp_on_logits(torch.nn.Module):
    def __init__(self):
        super(joint_prp_on_logits, self).__init__()
 
    def forward(self, inputs, targets):
        device = inputs.device

        custom_lr_pos = -0.005
        custom_lr_neg = -0.005
        with torch.no_grad():
            _, n = inputs.shape
            k = targets.sum(1)
            pos_grad = custom_lr_pos * (-1)
            neg_grad = custom_lr_neg * k / (n - k)

        # compute the positive loss
        pos_logits = targets*inputs         # (batch_size * label_num)
        pos_logits_sum = pos_logits.sum(1)  # (batch_size)
        loss_p = (-1)*pos_grad*pos_logits_sum

        # compute the negative loss
        neg_logits = (1-targets)*inputs     # (batch_size * label_num)
        neg_logits_sum = neg_logits.sum(1)  # (batch_size)
        loss_n = (-1)*neg_grad*neg_logits_sum

        # compute loss
        loss = loss_p + loss_n
        loss = torch.mean(loss)
        
        return loss


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
    def __init__(self, use_weighting=True):
        super(log_prp_Loss, self).__init__()
        self.use_weighting = use_weighting
 
    def forward(self, inputs, targets, ispositive=True, multiplicities=None, debug=False):

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

        if self.use_weighting:
            # input specific coefficient
            coefficient = (1.0 - probs.sum(dim=1))
            loss = coefficient.detach() * loss


        loss = torch.mean(loss)
        
        return loss

class bi_prp_loss(torch.nn.Module):
    def __init__(self, use_weighting=True):
        super(bi_prp_loss, self).__init__()
        self.use_weighting = use_weighting

    def forward(self, inputs, targets, debug=False):


        device = inputs.device
        input_sm = F.softmax(inputs, dim=1)

        # monitor logits
        if debug:
            logits = targets*inputs
            print("avg pos logit value:", torch.mean(logits).detach())
            print("avg logit value:", torch.mean(inputs).detach())

        
        pos_logits = inputs * targets
        pos_loss = (-1 * pos_logits).sum(dim=1)

        neg_logits = inputs * (1-targets)
        k = targets.sum(1, keepdims=True)
        k_neg = (1-targets).sum(1, keepdims=True)
        neg_weight = (k/(k_neg + EPS)).detach()
        neg_loss = (neg_weight * neg_logits).sum(dim=1)
        loss = neg_loss + pos_loss

        if self.use_weighting:
            # input specific coefficient
            coefficient = 1.0 - (input_sm * targets).sum(dim=1)
            # coefficient = torch.maximum(torch.tensor(0.0), coefficient - 0.3)
            loss = coefficient.detach() * loss

        loss = loss.mean()
        return loss

class bi_prp2_loss(torch.nn.Module):
    def __init__(self):
        super(bi_prp2_loss, self).__init__()
 
    def forward(self, inputs, targets):
        logprobs = F.log_softmax(inputs, dim=1)

        pos_logprobs = logprobs * targets
        neg_logprobs = logprobs * (1-targets)
        
        k = targets.sum(1, keepdims=True)
        k_neg = (1-targets).sum(1, keepdims=True)
        k_neg = torch.maximum(torch.tensor(1.0), k_neg)

        loss = (-1.0 * pos_logprobs + k / k_neg * neg_logprobs).sum(dim=1)

        # input specific coefficient
        probs = F.softmax(inputs, dim=1)
        coefficient = (probs * (1-targets)).sum(dim=1)
        loss = coefficient.detach() * loss

        loss = loss.mean()
        return loss
    
class bi_prp_nll_loss(torch.nn.Module):
    def __init__(self):
        super(bi_prp_nll_loss, self).__init__()
 
    def forward(self, inputs, targets, debug=False):


        device = inputs.device
        probs = F.softmax(inputs, dim=1)
        pos_probs = targets*probs

        # monitor logits
        if debug:
            logits = targets*inputs
            print("avg pos logit value:", torch.mean(logits).detach())
            print("avg logit value:", torch.mean(inputs).detach())

        
        pos_logits = inputs * targets
        pos_loss = (-1 * pos_logits).sum(dim=1)

        neg_logits = inputs * (1-targets)
        k = targets.sum(1, keepdims=True)
        k_neg = (1-targets).sum(1, keepdims=True)
        neg_weight = (k/(k_neg + EPS)).detach()
        neg_loss = (neg_weight * neg_logits).sum(dim=1)
        bi_prp_loss = neg_loss + pos_loss

        # input specific coefficient
        coefficient = 1.0 - pos_probs.sum(dim=1)
        # coefficient = torch.maximum(torch.tensor(0.0), coefficient - 0.3)
        bi_prp_loss = coefficient.detach() * bi_prp_loss


        # weight between nll_loss and bi_prp_loss
        nll_loss = - probs.sum(dim=1)
        nll_weight = pos_probs.sum(dim=1) ** 1
        nll_weight = nll_weight.detach()
        loss = nll_weight * nll_loss + (1-nll_weight) * bi_prp_loss

        loss = loss.mean()
        return loss
    
class nll_loss(torch.nn.Module):
    def __init__(self):
        super(nll_loss, self).__init__()
 
    def forward(self, inputs, targets):
        device = inputs.device
        input_sm = F.softmax(inputs, dim=1)

        probs = targets*input_sm
        sumprobs = probs.sum(1)

        # invprob = torch.maximum(torch.FloatTensor([EPS]).expand_as(sumprobs).to(device), 1.0 - sumprobs)

        # log_n = torch.log(invprob)
        # loss = torch.mean(log_n)
        loss = - (torch.log(sumprobs)).mean()
        total_probs = torch.sum(sumprobs)
        
        return loss, total_probs

class democracy_loss(torch.nn.Module):
    def __init__(self):
        super(democracy_loss, self).__init__()
 
    def forward(self, inputs, targets):
        device = inputs.device
        logprobs = F.log_softmax(inputs, dim=1)
        loss = -(logprobs * targets).sum(dim=1)
        loss = loss.mean()
        return loss

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


def lws_loss(outputs, partialY, confidence, index, lw_weight, lw_weight0, epoch_ratio, debug=False):
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
