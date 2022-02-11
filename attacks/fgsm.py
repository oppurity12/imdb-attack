import torch
import torch.nn as nn
import torch.optim as optim


def fgsm(model, test_loader, epsilon, device):
    criterion = nn.CrossEntropyLoss()
    adv_accuracy = 0
    model.train()
    for batch in test_loader:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환

        x = model.get_embed_vector(x).detach().cpu().clone()
        x = x.to(device)
        x.requires_grad = True
        logit = model.forward2(x)
        model.zero_grad()
        loss = criterion(logit, y)
        grad = torch.autograd.grad(
            loss, x, retain_graph=False, create_graph=False)[0]

        adv_x = x + epsilon*grad.sign()

        adv_logit = model.forward2(adv_x)
        adv_accuracy += (torch.argmax(adv_logit, dim=1)
                         == y).float().mean().item()

    adv_accuracy /= len(test_loader)
    return adv_accuracy * 100
