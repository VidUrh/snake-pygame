import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


MODEL_FOLDER_PATH = './model'


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        file_name = "model.pth"
        if os.path.exists(os.path.join(MODEL_FOLDER_PATH, file_name)):
            self.load_state_dict(torch.load(os.path.join(MODEL_FOLDER_PATH, file_name)))
            self.eval()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def save(self, file_name="model.pth"):
        MODEL_FOLDER_PATH = './model'
        if not os.path.exists(MODEL_FOLDER_PATH):
            os.makedirs(MODEL_FOLDER_PATH)
        file_name = os.path.join(MODEL_FOLDER_PATH, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # 1: predicted Q values with the current state
        pred = self.model(state)

        # 2: Q_new + gamma * max(next_predicted Q value)
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
