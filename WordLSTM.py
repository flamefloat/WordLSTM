import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict = {n: i for i, n in enumerate(char_arr)}
number_dict = {i: w for i, w in enumerate(char_arr)}
n_class = len(word_dict) # number of class(=number of vocab)

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

# TextLSTM Parameters
n_step = 3
hidden_size = 64

def make_batch(seq_data):
    input_batch, target_batch = [], []
    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]] # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]] # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)

#a, b = make_batch(seq_data)
#print(a.size(), b.size())

class WordLSTM(nn.Module):
    def __init__(self, n_class, hidden_size):
        super(WordLSTM, self).__init__()
        self.n_class = n_class
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.n_class, self.hidden_size, batch_first = True)
        self.fc = nn.Linear(self.hidden_size, self.n_class)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :] #最后时刻的输出
        output = self.fc(x)
        return output

model = WordLSTM(n_class, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
input_batch, target_batch = make_batch(seq_data)
for epoch in range(1000):
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

inputs = [sen[:3] for sen in seq_data]
predict = model(input_batch).data.max(1, keepdim=True)[1] # (max_value, max_position)
print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])
