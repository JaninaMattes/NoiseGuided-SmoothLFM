# --------------------------------
# For testing
# --------------------------------
class DummyDecoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=1024):
        super(DummyDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 32 * 32),
            nn.Unflatten(1, (in_channels, 32, 32))
        )
        self.initialize_weights()

    def forward(self, z):
        x = self.decoder(z)
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
    z = torch.randn(16, 1024).to(device)

    Test dummy decoder
    decoder = DummyDecoder(in_channels=in_channels, latent_dim=latent_dim).to(device)
    out = decoder(z)
    
    print(f"Decoder: {decoder}")
    print(f"Decoder Params: {count_trainable_parameters(decoder):,}")
    print(F"Decoder output shape: {out.shape}")