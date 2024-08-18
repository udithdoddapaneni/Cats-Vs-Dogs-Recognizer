import torch
from torch import nn
from torch import optim
from PIL import Image
import PIL
import numpy as np
from torchvision.transforms import Normalize
from warnings import filterwarnings

filterwarnings("ignore")

NORMALIZE = Normalize(
mean=(0.4883098900318146, 0.4550710618495941, 0.41694772243499756),
std =(0.25926217436790466, 0.25269168615341187, 0.25525805354118347)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) # 6 x 128 x 128
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.maxpool = nn.MaxPool2d(kernel_size=4) # 6 x 32 x 32

        class Block(nn.Module):
            def __init__(this, features):
                super().__init__()
                this.num_featues = features
                this.conv1 = nn.Conv2d(in_channels=this.num_featues, out_channels=this.num_featues, kernel_size=3, padding=1)
                this.conv2 = nn.Conv2d(in_channels=this.num_featues, out_channels=this.num_featues, kernel_size=3, padding=1)
                this.bn = nn.BatchNorm2d(num_features=this.num_featues)
            def forward(this, x):
                y = torch.relu(this.bn(this.conv1(x)))
                y = this.bn(this.conv2(y))
                return torch.relu(this.bn(x + y))
            
            __call__ = forward

        self.blocks1 = nn.ModuleList([Block(6) for i in range(3)]) # bn1 works
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, padding=1) # 24 x 32 x 32
        self.bn2 = nn.BatchNorm2d(num_features=24)
        # use self.maxpool again -----------------> # 24 x 8 x 8

        self.blocks2 = nn.ModuleList([Block(24) for i in range(3)])

        self.avg_pool = nn.AvgPool2d(kernel_size=2) # 24 x 4 x 4
        self.dropout = nn.Dropout()
        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.l1 = nn.Linear(in_features=24*4*4, out_features=32)
        self.l2 = nn.Linear(in_features=32, out_features=2)

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr, weight_decay=0.00001)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        for block in self.blocks1:
            x = block(x)
        x = self.maxpool(torch.relu(self.bn2(self.conv2(x))))
        for block in self.blocks2:
            x = block(x)
        x = self.avg_pool(x)
        x = torch.relu(self.l1(x.reshape(1, 24*4*4)))
        x = torch.sigmoid(self.l2(x))

        return x
    
    def predict(self, im: Image.Image | np.ndarray | torch.Tensor):
        with torch.no_grad():
            im = Image.fromarray(np.array(im))
            im = im.resize((128,128))
            im = NORMALIZE((torch.tensor([np.array(im)])/255.0).permute(dims=(0,3,1,2)))
            im = im.to(device)
            im = self.forward(im)
            label = im[0].argmax()
            if label.item():
                return "Dog"
            return "Cat"

MODEL = Model(lr=0.001)
MODEL.to(device)
def LoadModel(MODEL:Model):
    MODEL.load_state_dict(torch.load("Model/weights.pth"))


LoadModel(MODEL)