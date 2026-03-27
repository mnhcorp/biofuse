import torch

from biofuse.models.biofuse_model import BioFuseModel


def test_concat_supports_mixed_model_dimensions_without_projection():
    model = BioFuseModel(
        ["BioMedCLIP", "rad-dino"],
        fusion_method="concat",
        projection_dim=0,
    )

    output = model([
        torch.randn(2, 512),
        torch.randn(2, 768),
    ])

    assert output.shape == (2, 1280)


def test_projected_mean_uses_registered_model_dimensions():
    model = BioFuseModel(
        ["BioMedCLIP", "rad-dino"],
        fusion_method="mean",
        projection_dim=256,
    )

    output = model([
        torch.randn(2, 512),
        torch.randn(2, 768),
    ])

    assert output.shape == (2, 256)
