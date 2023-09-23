import os
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_fixer(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def index_preprocessing(way, shot, query):
    if shot == query :
        adaptation_indices = np.zeros(way*shot*2, dtype=bool)
        adaptation_indices[np.arange(way*shot)*2] = True

    elif shot != query :
        adaptation_indices = np.zeros(way*(shot+query), dtype=bool)
        adaptation_indices[np.arange(way*(shot+query))] = True
        for j in range(0, way*(shot+query), shot+query):
            for i in range(shot, shot+query):
                i += j
                adaptation_indices[i] = False

    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    return adaptation_indices, evaluation_indices


def knowledge_distillation_loss(student_logit, teacher_logit, T):
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_logit/T, dim=1), 
                                                  F.softmax(teacher_logit/T, dim=1)) * (T*T)
    return kd_loss


def confidence_interval(scores):
    # where df >= 1000  
    std = math.sqrt(np.var(scores))
    N = len(scores)
    ci = {
        "99%" : 2.576 * std/math.sqrt(N),
        "95%" : 1.960 * std/math.sqrt(N),
        "90%" : 1.645 * std/math.sqrt(N)
    }
    return ci


class MetaWeights(nn.Module):
    def __init__(self):
        super(MetaWeights, self).__init__()
        
        self.linear = nn.Linear(in_features=2, out_features=1, bias=False)
        torch.nn.init.constant_(self.linear.weight.data, 0.5)

    def forward(self, l1, l2):
        loss = torch.tensor([l1,l2])
        weights = self.linear(loss)
        return weights
