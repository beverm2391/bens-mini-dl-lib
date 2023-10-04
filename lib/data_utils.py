from pathlib import Path
import gzip
import pickle
import requests

def get_mnist(data_path: str):
    data_path = Path(data_path)
    path = data_path / "mnist"
    path.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (path / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (path / FILENAME).open("wb").write(content)

    with gzip.open((path / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_valid, y_valid