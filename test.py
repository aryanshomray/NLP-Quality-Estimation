import torch
from model import MainModel
from dataloader import testDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


dataset = testDataset()

model = MainModel(10310, 15439).cuda()
model.load_state_dict(torch.load("model_lstm.pth"))

total = 0
correct = 0

outs = []
preds = []

with torch.no_grad():
    for idx, data in tqdm(enumerate(dataset)):

        outputs = model(data["source"].unsqueeze(0).cuda(), data["target"].unsqueeze(0).cuda(),
                        data["alignment"].unsqueeze(0).cuda()).view(-1).cpu()
        # print(outputs.numpy() > 0.5)
        ans = np.sum((outputs.numpy() > 0.5) == data["predictions"].numpy())
        outs.extend((outputs.numpy() > 0.5))
        preds.extend(data["predictions"].numpy())
        # print(ans, outputs.size()[0])
        total += outputs.size()[0]
        correct += ans
        # print((outputs.numpy() > 0.5),"\n",data["predictions"].numpy())
print((outputs.numpy() > 0.5),"\n",data["predictions"].numpy())
print(f"Accuracy: {(correct/total)*100}")
print(f"F-1 Score(Macro): {f1_score(preds, outs, average='macro')*100}")
print(f"F-1 Score(Micro): {f1_score(preds, outs, average='micro')*100}")
print(f"F-1 Score(Weighted): {f1_score(preds, outs, average='weighted')*100}")

