# SCUNet CoreML

Core ML wrapper for SCUNet real-image denoising on Apple Silicon.

The Platypus path intentionally mirrors the existing `scunet_helper.py`:

1. keep the input in linear RGB;
2. apply Platypus `log1p_tonemap_forward`;
3. run SCUNet on 448px tiles with 64px overlap;
4. apply `log1p_tonemap_inverse`;
5. restore broad low-frequency color/tone from the original image.

Setup from Platypus:

```bash
cd /Users/uniuyuni/PythonProjects/platypus
git clone https://github.com/uniuyuni/SCUNet_CoreML.git SCUNet_CoreML
cd SCUNet_CoreML
pixi run setup-assets
pixi run install-platypus-helper
```

This installs `helpers/scunet_coreml_helper.py` into Platypus.

The default tile size is `32 * 14 = 448`, with overlap `32 * 2 = 64`.
