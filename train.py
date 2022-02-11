import torch
import torch.nn as nn
import torch.optim as optim

from loader import imbda_loader
from model import IMDBModel

from rnn_models.gru_alpha import GRUModelAlpha
from rnn_models.ssp_gru import SSRGRUModel
from rnn_models.ssp2_gru import SSP2GRUModel
from rnn_models.ssp3_gru import SSP3GRUModel

from attacks.fgsm import fgsm

import pandas as pd

import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, val_loader):
    model.eval()
    corrects, total_loss = 0, 0
    criterion = nn.CrossEntropyLoss()
    for batch in val_loader:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = criterion(logit, y)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def test(model, test_loader):
    model.eval()
    corrects = 0
    for batch in test_loader:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        logit = model(x)
        corrects += (logit.max(1)[1].view(y.size()
                                          ).data == y.data).sum().item()
    size = len(test_loader.dataset)
    avg_accuracy = 100.0 * corrects / size
    return avg_accuracy


def train(args):
    train_loader, val_loader, test_loader, vocab_size = imbda_loader(
        args.batch_size)

    rnn_model_lists = {
        "gru": nn.GRU,
        "lstm": nn.LSTM,
        "ssp_gru": SSRGRUModel,
        "ssp2_gru": SSP2GRUModel,
        "ssp3_gru": SSP3GRUModel,
        "gru_alpha": GRUModelAlpha
    }

    rnn_model = rnn_model_lists[args.model_type]

    model = IMDBModel(
        rnn_model,
        args.hidden_dim,
        vocab_size,
        args.embed_dim,
        args.num_layers,
        args.num_classes,
        args.lr,
        args.alpha).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    loss_lists = []
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    time0 = time.time()
    for epoch in range(args.training_epochs):
        model.train()
        for b, batch in enumerate(train_loader):
            x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
            y.data.sub_(1)  # 레이블 값을 0과 1로 변환
            optimizer.zero_grad()

            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()

            loss_lists.append(loss)
        avg_loss, avg_accuracy = evaluate(model, val_loader)

        print(
            f"epoch: {epoch + 1} val_loss: {avg_loss:.3f} val_accuracy: {avg_accuracy:.3f} time: {(time.time() - time0)/60:.3f}분")

    test_accuracy = test(model, test_loader)
    print(f"test_accuracy: {test_accuracy:.3f}")

    adv_accuracy = fgsm(model, test_loader,
                        epsilon=args.epsilon, device=DEVICE)
    print(f"adv_accuracy: {adv_accuracy:.3f}")

    results = {"test_accuracy": [test_accuracy],
               "adv_accuracy": [adv_accuracy]}
    results = pd.DataFrame(results)
    results.to_csv(args.save_path)

    return model, loss_lists
