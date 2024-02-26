_base_ = './deeplabv3plus_r50-d8_769x769_80k_pascal_synth.py'

pretrained_resnet101v1c = '/gfs-ssd/project/uss/pre_trained_models/ResNetv1c/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth'
model = dict(
    pretrained=None,
    backbone=dict(init_cfg=dict(
                    type='Pretrained',
                    checkpoint=pretrained_resnet101v1c,
                    prefix='backbone.'
                    ),
                  depth=101,
                 ),
    decode_head=dict(align_corners=True, num_classes=19+6),
    auxiliary_head=dict(align_corners=True, num_classes=19+6),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
)
