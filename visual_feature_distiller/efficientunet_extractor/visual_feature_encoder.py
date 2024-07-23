from .layers import *

def get_feature_encoder(config):
    teacher_type = config['image_processing'][0]['type']
    teacher_insize = config['image_processing'][0]['args']['image_insize']
    teacher_insize = (teacher_insize[1], teacher_insize[0]) # (H, W)
    teacher_outsize = None
    teacher_channels = None
    feature_encoder_name = config['feature_encoder_name']
    
    if teacher_type == 'dino':
        assert teacher_insize[0]%14 == 0 and teacher_insize[1]%14 == 0, 'DINO input size must be divisible by 14'
        teacher_outsize = (teacher_insize[0]//14, teacher_insize[1]//14) # (H, W)
        dino_type = config['image_processing'][0]['args']['dino_type']
        if dino_type == 'dinov2_vitg14':
            teacher_channels = 1536
        elif dino_type == 'dinov2_vitb14':
            teacher_channels = 768
        else:
            raise ValueError(f'Invalid DINO type: {dino_type}')
    elif teacher_type == 'radio':
        assert teacher_insize[0]%16 == 0 and teacher_insize[1]%16 == 0, 'Radio input size must be divisible by 32'
        teacher_outsize = (teacher_insize[0]//16, teacher_insize[1]//16)
        teacher_channels = 1280
    else:
        raise ValueError(f'Invalid teacher type: {teacher_type}')

    if feature_encoder_name == 'CnnEncoder':
        return CnnEncoder(teacher_outsize, teacher_channels)
    elif feature_encoder_name == 'CnnSimpleEncoder':
        return CnnSimpleEncoder(teacher_outsize, teacher_channels)
    elif feature_encoder_name == 'MaxPoolEncoder':
        return MaxPoolEncoder(teacher_outsize, teacher_channels)
    elif feature_encoder_name == 'InterpolateAreaEncoder':
        return InterpolateAreaEncoder(teacher_outsize, teacher_channels)
    elif feature_encoder_name == 'InterpolateBilinearEncoder':
        return InterpolateBilinearEncoder(teacher_outsize, teacher_channels)
    elif feature_encoder_name == 'InterpolateAreaSimpleEncoder':
        return InterpolateAreaSimpleEncoder(teacher_outsize, teacher_channels)
    else:
        raise ValueError(f'Invalid feature encoder name: {feature_encoder_name}')

class CnnEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, teacher_channels, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        x = self.network(x)
        return x

class CnnSimpleEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, teacher_channels, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        x = self.network(x)
        return x
    
class MaxPoolEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1536, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        x = self.network(x)
        return x
    
class InterpolateAreaEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            InterpolateArea(size=teacher_outsize),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, teacher_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        x = self.network(x)
        return x

class InterpolateAreaSimpleEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            InterpolateArea(size=teacher_outsize),
            nn.Conv2d(16, teacher_channels, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        x = self.network(x)
        return x

class InterpolateBilinearEncoder(nn.Module):
    def __init__(self, teacher_outsize, teacher_channels):
        super().__init__()
        
        self.network = nn.Sequential(
            InterpolateBilinear(size=teacher_outsize),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, teacher_channels, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        x = self.network(x)
        return x