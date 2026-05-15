import os

import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import src.cnn as cnn


def _make_image(path, size=(16, 16), color=(128, 64, 32)):
    Image.new("RGB", size, color).save(path)


def test_train_df_filters_supported_extensions(tmp_path):
    class_a = tmp_path / "class_a"
    class_a.mkdir()
    _make_image(class_a / "img1.jpg")
    (class_a / "notes.txt").write_text("not an image", encoding="utf-8")

    result = cnn.train_df(str(tmp_path))

    assert list(result.columns) == ["Class Path", "Class"]
    assert len(result) == 1
    assert result.iloc[0]["Class"] == "class_a"
    assert result.iloc[0]["Class Path"].endswith("img1.jpg")


def test_test_df_filters_supported_extensions(tmp_path):
    class_b = tmp_path / "class_b"
    class_b.mkdir()
    _make_image(class_b / "img2.PNG")
    (class_b / "readme.md").write_text("metadata", encoding="utf-8")

    result = cnn.test_df(str(tmp_path))

    assert list(result.columns) == ["Class Path", "Class"]
    assert len(result) == 1
    assert result.iloc[0]["Class"] == "class_b"
    assert result.iloc[0]["Class Path"].endswith("img2.PNG")


def test_tumor_dataset_returns_tensor_and_label(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path)

    df = pd.DataFrame([{"Class Path": str(image_path), "Class": "x", "Label": 2}])
    dataset = cnn.TumorDataset(df, transform=None)
    image, label = dataset[0]

    assert isinstance(image, Image.Image)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert int(label.item()) == 2


def test_tumor_dataset_raises_for_missing_file():
    df = pd.DataFrame([{"Class Path": "/no/such/file.jpg", "Class": "x", "Label": 1}])
    dataset = cnn.TumorDataset(df)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_train_and_evaluate_metric_contracts():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 3, (8,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    eval_loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    train_loss, train_acc = cnn.train_one_epoch(model, train_loader, criterion, optimizer, device)
    eval_loss, eval_acc = cnn.evaluate(model, eval_loader, criterion, device)

    for loss in (train_loss, eval_loss):
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
    for acc in (train_acc, eval_acc):
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


def test_build_model_respects_class_count():
    num_classes = 4
    model = cnn.build_model(num_classes, weights=None)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, num_classes)
