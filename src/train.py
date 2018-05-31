"""Train a SegNet model"""

from model import SigNet
import torch
import time


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 3
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9


model = SigNet(input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS)

# TODO - create dataloaders
train_data = None
val_data = None

criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)



model.train()

for epoch in range(NUM_EPOCHS):
    for i, (input_img, target) in enumerate(train_data):
        t_start = time.time()

        input_tensor = torch.autograd.Variable(input_img)
        target_tensor = torch.autograd.Variable(target)

        predicted_tensor = model(input_var)

        loss = criterion(predicted_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_f = loss.float()
        prediction_f = predicted_tensor.float()
        
        delta = time.time() - t_start

        print(f"""Epoch #{i+1}
              Loss = {loss_f}
              Time = {delta}secs
              """)
