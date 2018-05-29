"""Train a SegNet model"""

from model import SigNet
import torch
import time


# Constants
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_EPOCHS = 10

model = SigNet(input_channels=NUM_CHANNELS, output_channels=NUM_CLASSES)

# TODO - create dataloaders
train_data = None
val_data = None
# TODO - choose loss function
loss = None
# TODO - choose optimiser
optimiser = None



model.train()

for epoch in range(NUM_EPOCHS):
    for i, (input_img, target) in enumerate(train_data):
        t_start = time.time()

        input_tensor = torch.autograd.Variable(input_img)
        target_tensor = torch.autograd.Variable(target)

        predicted_tensor = model(input_var)

        loss = optimiser(predicted_tensor, target_tensor)

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
