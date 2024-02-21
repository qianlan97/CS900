import torch
import torch.nn as nn
import time

# Define the simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define input size and create a sample input
input_size = 28 * 28  # Input image size
hidden_size1 = 128
hidden_size2 = 128
output_size = 10
batch_size = 1  # One batch
input_data = torch.rand(batch_size, input_size)  # Random input data for one image

# Create an instance of the model
model = SimpleModel(input_size, hidden_size1, hidden_size2, output_size)

# Warm-up to ensure any initializations are done (optional but recommended)
_ = model(input_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
model = model.to(device)
input_data = input_data.to(device)

# Measure execution time
start_time = time.time()
output = model(input_data)
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
