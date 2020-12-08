import torch
import torch.nn as nn
import torch.nn.functional as F


def align_vector(data, target, alignment, align_type):
    A = torch.zeros(alignment.size(), device="cuda:0")
    output = torch.zeros(target.size(), device="cuda:0", dtype=torch.float32)
    if align_type == "mid":
        A = alignment
        output = target
    elif align_type == "left" and target.size(1) > 1:
        A[:, :-1, :] = alignment[:, 1:, :]
        output[:, :-1, :] = target[:, 1:, :]
    elif align_type == "right" and target.size(1) > 1:
        A[:, 1:, :] = alignment[:, :-1, :]
        output[:, 1:, :] = target[:, :-1, :]
    else:
        pass
    A = A.float()
    data_align = torch.bmm(data, A)
    alignment_cnt = torch.sum(alignment, dim=1)
    alignment_cnt[alignment_cnt == 0] = 1
    data_align /= alignment_cnt[:, None, :]
    data_align = data_align.permute(0, 2, 1)
    return data_align, output


class NN(nn.Module):

    def __init__(self, input_size):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, 400)
        self.linear_2 = nn.Linear(400, 400)
        self.GRU_1 = nn.LSTM(400, 200, num_layers=1, bidirectional=True)
        self.linear_3 = nn.Linear(400, 400)
        self.linear_4 = nn.Linear(400, 400)
        self.GRU_2 = nn.LSTM(400, 100, num_layers=1, bidirectional=True)
        self.linear_5 = nn.Linear(400, 400)
        self.linear_6 = nn.Linear(400, 50)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.GRU_1(x)[0]
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = F.relu(self.linear_5(x))
        x = F.relu(self.linear_6(x))
        return x


class MainModel(torch.nn.Module):

    def __init__(self, num_words1, num_words2):
        super(MainModel, self).__init__()

        embedding_size: int = 64
        conv_size: int = 64
        self.embedding1 = nn.Embedding(num_words1, embedding_size)
        self.embedding2 = nn.Embedding(num_words2, embedding_size)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(embedding_size * 6, conv_size, 1, 1, 0),
                nn.Conv1d(embedding_size * 6, conv_size, 3, 1, 1),
                nn.Conv1d(embedding_size * 6, conv_size, 5, 1, 2),
                nn.Conv1d(embedding_size * 6, conv_size, 7, 1, 3)
            ]
        )
        self.linear_layers = NN(conv_size * len(self.conv_layers))
        self.linear = nn.Linear(50, 1, bias=True)

    def forward(self, source, target, alignment):
        source = self.embedding1(source)
        target = self.embedding2(target)

        source = source.permute(0, 2, 1)
        source_align_left, target_left = align_vector(source, target, alignment, align_type="left")
        source_align_centre, target_centre = align_vector(source, target, alignment, align_type="mid")
        source_align_right, target_right = align_vector(source, target, alignment, align_type="right")

        features = torch.cat([source_align_left, source_align_centre, source_align_right, target_left, target_centre,
                              target_right], dim=2).permute(0, 2, 1)
        conv_out = torch.cat(
            [
                F.relu(self.conv_layers[0](features)),
                F.relu(self.conv_layers[1](features)),
                F.relu(self.conv_layers[2](features)),
                F.relu(self.conv_layers[3](features))
            ], dim=1
        ).permute(0, 2, 1)
        linear_out = self.linear_layers(conv_out)
        output = self.linear(linear_out)
        output = torch.sigmoid(output)
        return output


if __name__ == "__main__":
    MainModel(512)
