# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from pytouch.handlers import ImageHandler
from digit_interface.digit import Digit
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time

# Set id for your DIGIT e.g. D20221
DIGIT_ID = "YOUR_DIGIT_ID"

digit = Digit(DIGIT_ID, "Left Gripper")
digit.connect()
# Change LED illumination intensity
digit.set_intensity(Digit.LIGHTING_MIN)
time.sleep(1)
digit.set_intensity(Digit.LIGHTING_MAX)
# Change DIGIT resolution to QVGA
qvga_res = Digit.STREAMS["QVGA"]
digit.set_resolution(qvga_res)
# Change DIGIT FPS to 30fps
fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
digit.set_fps(fps_30)

# Define Net which should be the same as the one defined in model.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp1 = nn.Linear(3*224*224, 500)
        self.mlp2 = nn.Linear(500, 128)
        self.mlp3 = nn.Linear(128, 2)

        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.reshape(-1, 3*224*224)
        x = self.bn1(F.relu(self.mlp1(x)))
        x = self.bn2(F.relu(self.mlp2(x)))
        x = self.mlp3(x)
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = Net().to(device) # if you want to use GPU
model = Net()

model.load_state_dict(torch.load("~/PATH/TO/YOUR/MODEL/DIR.pth",map_location=torch.device('cpu')))
# model.load_state_dict(torch.load("~/PATH/TO/YOUR/MODEL/DIR.pth",map_location=torch.device('cuda:0'))) # if you want to use GPU
model = model.eval()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

while True:
    import time
    start = time.time()
    frame = digit.get_frame()[:, :, ::-1]
    digit.save_frame("test.png")
    source = ImageHandler("test.png")   
    source = transform(source.img)
    output = model(source)
    _, pred = torch.max(output, 1)
    print("Predicted: ", pred.item())
    cv2.imshow("Frame", frame)
    time = time.time() - start
    print(f"Time: {time}")
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break





