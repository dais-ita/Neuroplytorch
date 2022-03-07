import torch 
from torch import nn 
import torch.nn.functional as F 
import numpy as np 

def fetch_default_device() -> torch.device:
    """ Fetch default pytorch device (cuda or cpu)

    :return: Pytorch device object
    :rtype: torch.device
    """
    default_device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    return default_device

def get_edl_losses() -> list:
    """ Return the list of edl loss function strings

    :return: list of edl loss function strings
    :rtype: list
    """
    return ['edl_mse_loss', 'edl_mse_0_loss', 'edl_log_loss', 'edl_log_0_loss', 'edl_digamma_loss', 'edl_digamma_0_loss', 'edl_log_1_loss']

def get_loss_fct(loss_str: str):
    """ Get loss function from loss function string

    :param loss_str: Name of loss function as string
    :type loss_str: str

    :return: Callable loss function
    """
    loss_fct = []
    if loss_str=='MSELoss': loss_fct = nn.MSELoss() 
    elif loss_str=='CrossEntropyLoss': loss_fct = nn.CrossEntropyLoss() 
    elif loss_str=='edl_log_loss': loss_fct = edl_log_loss
    elif loss_str=='edl_mse_loss': loss_fct = edl_mse_loss
    elif loss_str=='edl_digamma_loss': loss_fct = edl_digamma_loss

    return loss_fct

def edl_accuracy(yTrue: torch.tensor, yPred: torch.tensor) -> torch.tensor:
    """ Get accuracy if using EDL loss function and labels

    :param yTrue: Ground truth label
    :type yTrue: torch.tensor
    :param yPred: Output of model 
    :type yPred: torch.tensor

    :return: Mean accuracy of yPred with yTrue as ground truth
    :type: torch.tensor
    """
    pred = torch.argmax(yPred, axis=1)
    truth = torch.argmax(yTrue, axis=1)
    match = torch.reshape(torch.Tensor([torch.equal(pred, truth)]), (-1,1))
    return torch.mean(match)

def res_to_dirichlet(ev: torch.tensor) -> tuple:
    """ Calculates belief vector and uncertainty value for the output of a model trained with EDL loss 

    :param ev: Output of model
    :type ev: torch.tensor

    :return: Tuple of (uncertainty values, belief vectors) - batched outputs stacked
    :rtype: tuple
    """
    uncerts, beliefs = [], [] 
    x = torch.split(ev, 1)
    for xx in x:
        alpha = xx+1
        S = torch.sum(alpha, axis=1)
        K = 2.0 
        uncerts.append(K/S) 
        beliefs.append(xx/S) 
    
    return torch.stack(uncerts), torch.stack(beliefs)
    #return uncert.cpu().detach().numpy(), belief.cpu().detach().numpy()

def combined_loss(y_true, reasoning_out, intermediate_out, omega_value):
    """ Loss function used in Neuroplex [NOTE: not yet fully implemented]

    :param y_true: Ground truth label
    :param reasoning_out: Output of reasoning layer
    :param intermediate_out: Output of perception layer
    :param omega_value: Omega value

    :return: Combined loss value
    """
    omega = omega_value
    mse_loss = F.mse_loss(reasoning_out, y_true)

    EPSILON = 1e-7
 
    t_vec = intermediate_out / (1 - intermediate_out + EPSILON)

    t_prod = torch.prod(1 - intermediate_out + EPSILON, axis=2)
    t_prod = t_prod[:,:,None]
    logic_loss = torch.sum(t_vec * t_prod, axis=2)
    logic_loss = torch.sum(logic_loss, axis=1)

    new_loss = omega * logic_loss + (1 - omega) * mse_loss
    raise NotImplementedError
    #return torch.mean(new_loss, dtype=torch.float32)

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = fetch_default_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = fetch_default_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss



# LAMBDA 0 LOSSES:

def edl_mse_0_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_0_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

def mse_0_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.tensor(0.0, dtype=torch.float32)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)

    return loglikelihood + kl_div

def edl_log_0_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_0_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_0_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.tensor(0.0, dtype=torch.float32)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_digamma_0_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_0_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_log_1_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = fetch_default_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_1_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_1_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.tensor(1.0, dtype=torch.float32)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div