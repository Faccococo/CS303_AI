import torch  # 命令行是逐行立即执行的
content = torch.load('data.pth')
print(content.keys())   # keys()
print(content['label'])
