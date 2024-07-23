from collections import OrderedDict
from .layers import *
from .utils import print_cuda_memory_usage

__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks

class EfficientUnetExtractor(nn.Module):
    '''
    3 x H x W
    --image_encoder-> n_channels x H/32 x W/32
    --unet_upsample-> out_channels x H/4 x W/4 (SHLD MAKE THIS adjustable)
    --feature_encoder-> teacher_channel x H_teacher x W_teacher
    '''
    def __init__(self, image_encoder, feature_encoder, out_channels=16):
        super().__init__()

        self.image_encoder = image_encoder
        
        self.upsample1 = upsample_two_conv(self.n_channels, self.size[0])
        self.double_conv1 = double_conv_k3(self.size[0] + self.size[0], self.size[0])
        self.upsample2 = upsample_two_conv(self.size[0], self.size[1])
        self.double_conv2 = double_conv_k3(self.size[1] + self.size[1], self.size[1])
        self.upsample3 = upsample_two_conv(self.size[1], self.size[2])
        self.double_conv3 = double_conv_k3(self.size[2] + self.size[2], self.size[2])

        # self.up_conv1 = up_conv_two(self.n_channels, 512)
        # self.double_conv1 = double_conv_k3(self.size[0] + 512, 512)
        # self.up_conv2 = up_conv_two(512, 128)
        # self.double_conv2 = double_conv_k3(self.size[1] + 128, 128)
        # self.up_conv3 = up_conv_two(128, 32)
        # self.double_conv3 = double_conv_k3(self.size[2] + 32, 32)
        
        self.final_conv = nn.Conv2d(self.size[2], out_channels, kernel_size=1) 
        
        ## Feature Encoder
        self.feature_encoder = feature_encoder
        
    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.image_encoder.name]

    @property
    def size(self):
        ''' dimension of output at each downsample level excluding last one at the end of backbone (nchannels)'''
        size_dict = {'efficientnet-b0': [80, 40, 24, 16, 3], 'efficientnet-b1': [80, 50, 24, 16, 3],
                     'efficientnet-b2': [88, 48, 24, 16, 3], 'efficientnet-b3': [96, 48, 32, 24, 3],
                     'efficientnet-b4': [112, 56, 32, 24, 3], 'efficientnet-b5': [128, 64, 40, 24, 3],
                     'efficientnet-b6': [144, 72, 40, 32, 3], 'efficientnet-b7': [160, 80, 48, 32, 3]}
        return size_dict[self.image_encoder.name]

    def forward(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.image_encoder, x)
        _, x = blocks.popitem()
        efficientnet_output = x

        # x = self.up_conv1(x)
        x = self.upsample1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        # x = self.up_conv2(x)
        x = self.upsample2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        # x = self.up_conv3(x)
        x = self.upsample3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        final_conv_output = self.final_conv(x)
        
        feature_encoder_output = self.feature_encoder(final_conv_output)
        return efficientnet_output, final_conv_output, feature_encoder_output
    
    def extract_features(self, x):
        input_ = x
        blocks = get_blocks_to_be_concat(self.image_encoder, x)
        _, x = blocks.popitem()
        efficientnet_output = x

        # x = self.up_conv1(x)
        x = self.upsample1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        # x = self.up_conv2(x)
        x = self.upsample2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        # x = self.up_conv3(x)
        x = self.upsample3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        final_conv_output = self.final_conv(x)
        
        return efficientnet_output, final_conv_output