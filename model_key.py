import torch


if __name__=='__main__':
    checkpoint = torch.load("results/train/20230111-151232-angular_tiny_quad_224-224/model_best.pth.tar", map_location='cpu')
    print(checkpoint.keys())
    # for k,v in checkpoint['state_dict'].items():
    #     print(k)
    # checkpoint['model'] = checkpoint['state_dict']
    # torch.save(checkpoint,'best_model2.pth.tar')