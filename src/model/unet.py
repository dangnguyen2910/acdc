import torch 
import torch.nn as nn


class Encoder(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3,3,3), 
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3,3,3), 
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
    
    
    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x
    
    
class Analysis(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.encoder1 = Encoder(1, 26)
        self.encoder2 = Encoder(26, 52)
        self.encoder3 = Encoder(52, 104)
        self.encoder4 = Encoder(104, 208)
        self.encoder5 = Encoder(208, 416)
        self.pool = nn.MaxPool3d((1,2,2))
        
    
    def forward(self, x): 
        x1 = self.encoder1(x)
        x1_pool = self.pool(x1)
        
        x2 = self.encoder2(x1_pool)
        x2_pool = self.pool(x2)

        x3 = self.encoder3(x2_pool)
        x3_pool = self.pool(x3)

        x4 = self.encoder4(x3_pool)
        x4_pool = self.pool(x4)

        x5 = self.encoder5(x4_pool)
               
        return [x1, x2, x3, x4, x5]


class Decoder(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(1,2,2), 
            stride=(1,2,2),   
        )
        
        self.conv1 = nn.Conv3d(
            in_channels=out_channels * 2, 
            out_channels=out_channels, 
            kernel_size=(3,3,3), 
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3,3,3), 
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU()
        

    def forward(self, x, x_encoder): 
        x = self.up(x) 
        x = torch.cat([x, x_encoder], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x
      
    
class Synthesis(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.decoder1 = Decoder(416, 208)
        self.decoder2 = Decoder(208, 104)
        self.decoder3 = Decoder(104, 52)
        self.decoder4 = Decoder(52, 26)
        
        self.up1 = nn.ConvTranspose3d(
            in_channels=104, 
            out_channels=52, 
            kernel_size=(1, 2, 2), 
            stride=(1,2,2),   
        )
        
        self.up2 = nn.ConvTranspose3d(
            in_channels=52, 
            out_channels=26, 
            kernel_size=(1, 2,2), 
            stride=(1,2,2),   
        )
        
    
    def forward(self, x, x_encoder): 
        x1 = self.decoder1(x, x_encoder[3]) 
        
        x2 = self.decoder2(x1, x_encoder[2]) 
        x3 = self.decoder3(x2, x_encoder[1]) 
        x4 = self.decoder4(x3, x_encoder[0]) 
        z = self.up2(x3 + self.up1(x2))
        
        return x4 + z
    


class UNet3D(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.analysis = Analysis()
        self.synthesis = Synthesis()
        
        self.conv_last = nn.Conv3d(
            in_channels=26, 
            out_channels=3, 
            kernel_size=(1,1,1),
        )
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self, x): 
        x_encoder = self.analysis(x)
        output = self.synthesis(x_encoder[4], x_encoder)
        
        output = self.conv_last(output)
        output = self.sigmoid(output)
        return output
        
        

if __name__ == "__main__": 
    model = UNet3D()
    print(model)
    
    tensor = torch.rand(1,1,10, 224, 224)
    output = model(tensor)
    print(output.size())