# --------------------------------
# For Testing
# --------------------------------
class DummyEncoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=1024):
        super(DummyEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * 32 * 32, latent_dim * 2),
        )

        self.initialize_weights()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    




if __name__ == "__main__":
    device = torch.device("cpu")
    in_channels = 4
    latent_dim = 1024
    ipt = torch.randn((16, 4, 32, 32)).to(device)

    # Test dummy encoder
    encoder = DummyEncoder(in_channels=in_channels, latent_dim=latent_dim).to(device)
    print(f"Input Shape: {ipt.shape}")
    print(f"Encoder: {encoder}")
    print(f"Paremeters: {count_trainable_parameters(encoder)}")