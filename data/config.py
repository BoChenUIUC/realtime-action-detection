# config.py
"""  SSD network configs

Original author: Ellis Brown, Max deGroot for VOC dataset
https://github.com/amdegroot/ssd.pytorch

"""
#SSD512 CONFIGS
v5 = {
    'feature_maps' : [(64,64), (32,32), (16,16), (8,8), (4,4), (2,2), (1,1)],

    'min_dim' : [512,512],

    'steps' : [(8,8), (16,16), (32,32), (64,64), (128,128), (256,256), (512,512)],

    'min_sizes' : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'max_sizes' : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'num_rotations' : 8,

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v5',

    'no_rotation' : False,
}

#SSD512 CONFIGS
v4 = {
    'feature_maps' : [(64,128), (32,64), (16,32), (8,16), (4,8), (2,4), (1,2)],

    'min_dim' : (512,1024),

    'steps' : [(8,8), (16,16), (32,32), (64,64), (128,128), (256,256), (512,512)],

    'min_sizes' : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'max_sizes' : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'num_rotations' : 8,

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v4',

    'no_rotation' : True,
}

#SSD512 CONFIGS
v3 = {
    'feature_maps' : [(64,128), (32,64), (16,32), (8,16), (4,8), (2,4), (1,3)],

    'min_dim' : (512,1024),

    'steps' : [(8,8), (16,16), (32,32), (64,64), (128,128), (256,256), (512,512)],

    'min_sizes' : [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'max_sizes' : [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'num_rotations' : 8,

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v3',

    'no_rotation' : True,
}

#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v2',
}

sph_v2 = {
    'feature_maps' : [(38,75), (19,38), (10,19), (5,10), (3,5), (1,3)],

    'min_dim' : (300,600),

    'steps' : [(8,8), (16,16), (32,32), (64,64), (100,120), (300,200)],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'num_rotations' : 8,

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'sph_v2',

    'no_rotation' : True, # inplane rotation
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 114, 168, 222, 276],

    'max_sizes' : [-1, 114, 168, 222, 276, 330],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
                        [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v1',
}
