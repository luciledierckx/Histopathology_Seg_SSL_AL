import torch 
import torch.nn as nn

## Code coming from https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.ReLU()  

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))   

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)     
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_class=1, input_size=64):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, input_size)
        self.e2 = encoder_block(input_size, input_size*2)
        self.e3 = encoder_block(input_size*2, input_size*4)
        self.e4 = encoder_block(input_size*4, input_size*8)         
        """ Bottleneck """
        self.b = conv_block(input_size*8, input_size*16)         
        """ Decoder """
        self.d1 = decoder_block(input_size*16, input_size*8)
        self.d2 = decoder_block(input_size*8, input_size*4)
        self.d3 = decoder_block(input_size*4, input_size*2)
        self.d4 = decoder_block(input_size*2, input_size)         
        """ Classifier """
        self.outputs = nn.Conv2d(input_size, n_class, kernel_size=1, padding=0)     
    
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)         
        """ Bottleneck """
        b = self.b(p4)         
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        """ Classifier """
        outputs = self.outputs(d4)        
        return outputs