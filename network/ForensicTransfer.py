import torch
import torch.nn as nn

class ForensicTransfer(nn.Module):
    def __init__(self, num_classes=4):
        super(ForensicTransfer, self).__init__()
        self.layer_d1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer_d2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_d3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_d4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer_d5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer_u5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer_u4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer_u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer_u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            #torch.nn.Dropout2d(p=0.2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.layer_u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            # nn.ReLU()
            nn.Tanh()
        )

        self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)  # b4-b5=2048

    def forward(self, x):
        latent = self.layer_d1(x)
        latent = self.layer_d2(latent)
        latent = self.layer_d3(latent)
        latent = self.layer_d4(latent)
        latent = self.layer_d5(latent)

        reconstruct = self.layer_u5(latent)
        reconstruct = self.layer_u4(reconstruct)
        reconstruct = self.layer_u3(reconstruct)
        reconstruct = self.layer_u2(reconstruct)
        reconstruct = self.layer_u1(reconstruct)

        clf = nn.AdaptiveAvgPool2d(1)(latent)
        clf = clf.view(clf.size(0), -1)
        clf = self.fc(clf)
        return clf, reconstruct


if __name__ == '__main__':
    model, image_size = ForensicTransfer(), 512
    # print(model)
    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))
