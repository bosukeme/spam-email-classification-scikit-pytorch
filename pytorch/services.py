import torch
import torch.nn as nn
import pickle
import json


with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/artifact.json", "r") as f:
    rnn_inputs = json.load(f)


class SpamRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out


input_dim = rnn_inputs["input_dim"]
hidden_dim = rnn_inputs["hidden_dim"]
output_dim = rnn_inputs["output_dim"]
num_layers = rnn_inputs["num_layers"]


model = SpamRNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("models/spam_model_rnn.pth"))


def predict_message(message: str):
    message_vec = vectorizer.transform([message]).toarray()
    message_tensor = torch.tensor(message_vec, dtype=torch.float)

    model.eval()
    with torch.inference_mode():
        output = model(message_tensor)
        prediction = torch.argmax(output, axis=1)
    return "Spam" if prediction.item() == 1 else "Not Spam"
