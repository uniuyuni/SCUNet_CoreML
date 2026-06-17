import torch

from scunet_coreml.model import build_scunet_color_real_psnr


def test_forward_shape():
    model = build_scunet_color_real_psnr(tile=448).eval()
    x = torch.zeros(1, 3, 448, 448)
    with torch.inference_mode():
        y = model(x)
    assert tuple(y.shape) == tuple(x.shape)
