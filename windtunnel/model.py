"""Small 2D U-Net, about 1-2M parameters. (Owner: Abhay; needs torch)

Input: normalised [N, 3, 64, 128] (sdf, mask, re_norm).
Output: normalised [N, 3, 64, 128] (u, v, p).
Normalisation, denormalisation and masking are added around this network
at export time so the ONNX graph takes raw inputs (see ABHAY.md).
"""
