import torch

from model import MainModel
from dataloader import trainDataset
from tqdm import tqdm

EPOCHS = 100


def pad_source(src, PAD):
    max_len = max(len(s) for s in src)
    for s in src:
        s += [0] * (max_len - len(s))
    return src, max_len


def pad_align(align):
    src_len = max(len(a) for a in align)
    trg_len = max(len(a[0]) for a in align)

    for a in align:
        a += [[0] * trg_len] * (src_len - len(a))
    return align, src_len, trg_len


def train():
    dataset = trainDataset()
    model = MainModel(dataset.vocab_size()[0], dataset.vocab_size()[1])
    #model.load_state_dict(torch.load("model2.pth"))
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             pin_memory=True)

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}/{EPOCHS}")
        losses = []

        for idx, data in tqdm(enumerate(dataloader)):
            outputs = model(data["source"].cuda(), data["target"].cuda(), data["alignment"].cuda())
            loss = torch.nn.functional.binary_cross_entropy(outputs.view(-1), data["predictions"].cuda().view(-1)
                                                            .float())
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        print(f"Mean Loss for Epoch: {epoch} is {sum(losses) / len(losses)}")
        torch.save(model.state_dict(), f"model_lstm.pth")


if __name__ == "__main__":
    train()
