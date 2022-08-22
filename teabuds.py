import os, yaml, random, math
from detectron2.data import DatasetCatalog, MetadataCatalog

# Dataset loading utilities
def register(name, path):
    """
    Register a tea leaves dataset.
    """
    items, meta = _load_dataset(path)
    _register_dataset(name, items, meta, 0.8)

def _load_dataset(dir):
    """
    Load downloaded dataset from given directory.

    The directory should contain a file "dataset.yaml"
    """
    with open(os.path.join(dir, "dataset.yaml"), "r") as file:
        dataset = yaml.safe_load(file)
    items = []
    for file in dataset["data"]:
        filepath = os.path.realpath(os.path.join(dir, file))
        if not os.path.exists(filepath):
            continue

        dict = dataset["data"][file]
        dict["file_name"] = filepath
        items.append(dict)
    # Shuffle the items
    random.shuffle(items)
    meta = {}
    if "categories" in dataset["metadata"]:
        thing_classes = list(
            map(
                lambda category: category["display_name"],
                dataset["metadata"]["categories"],
            )
        )
        meta["thing_classes"] = thing_classes
    if "points" in dataset["metadata"]:
        meta["keypoint_names"] = dataset["metadata"]["points"]
    if "lines" in dataset["metadata"]:
        meta["keypoint_connection_rules"] = dataset["metadata"]["lines"]
    meta["keypoint_flip_map"] = []
    return items, meta

def _register_dataset(name, items, meta={}, splits=0.8):
    """
    Register detectron2 data set.

    It will split the data into `train` and `val` set.
    """
    i = math.floor(len(items) * splits)
    train_items = items[:i]
    val_items = items[i:]

    train_name = f"{name}/train"
    val_name = f"{name}/val"
    try:
        DatasetCatalog.remove(train_name)
        DatasetCatalog.remove(val_name)
    except KeyError:
        pass
    DatasetCatalog.register(train_name, lambda: train_items)
    if len(val_items) > 0:
        DatasetCatalog.register(val_name, lambda: val_items)
    for k, v in meta.items():
        setattr(MetadataCatalog.get(train_name), k, v)
        setattr(MetadataCatalog.get(val_name), k, v)

register("tea_leaves", "datasets/tea_leaves")