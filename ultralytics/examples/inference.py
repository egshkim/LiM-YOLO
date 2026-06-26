import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import OBB, DualOBB

model = YOLO("path/to/best.pt")

metrics = model.val(
    data="your_dataset.yaml",
    imgsz=1024,
    batch=2,
    save_json=True,
)


# Deploy a trained LiM-YOLO-RB at LiM-YOLO cost.
# The reversible branch (the auxiliary mini-FPN and its detection head) is trained only and
# does not affect the main-branch predictions. Removing it restores the LiM-YOLO main path,
# so the deployed model matches LiM-YOLO in parameters, GFLOPs, and latency while keeping the
# weights learned with reversible-branch supervision.


def _from(layer):
    f = layer.f if isinstance(layer.f, list) else [layer.f]
    return [(j if j >= 0 else layer.i + j) for j in f]


def strip_reversible_branch(model):
    """Remove the reversible branch from a trained LiM-YOLO-RB model in place."""
    layers = list(model.model)
    head = layers[-1]
    assert isinstance(head, DualOBB), "expected a LiM-YOLO-RB (DualOBB) model"
    main_inputs = _from(head)[head.nl :]  # the head takes [aux..., main...]; keep the main half

    keep, stack = set(), list(main_inputs)
    while stack:  # walk back from the main inputs to the layers the main path needs
        j = stack.pop()
        if j < 0 or j in keep:
            continue
        keep.add(j)
        stack += _from(layers[j])
    keep = sorted(keep)
    remap = {old: new for new, old in enumerate(keep)}

    kept = []
    for old in keep:
        layer = layers[old]
        if isinstance(layer.f, list):
            layer.f = [(-1 if j == -1 else remap[j if j >= 0 else old + j]) for j in layer.f]
        else:
            layer.f = -1 if layer.f == -1 else remap[layer.f if layer.f >= 0 else old + layer.f]
        layer.i = remap[old]
        kept.append(layer)

    head.fuse()  # drop the auxiliary detection convs
    head.__class__ = OBB  # the main branch becomes a plain OBB head
    head.f = [remap[j] for j in main_inputs]
    head.i = len(kept)
    kept.append(head)

    model.model = torch.nn.Sequential(*kept)
    model.save = sorted({j for layer in kept for j in (layer.f if isinstance(layer.f, list) else [layer.f]) if j != -1})
    model.stride = head.stride
    return model


rb = YOLO("path/to/lim-yolo-rb.pt")
strip_reversible_branch(rb.model)
rb.save("lim-yolo-rb-deploy.pt")

deploy = YOLO("lim-yolo-rb-deploy.pt")
deploy.val(data="your_dataset.yaml", imgsz=1024, batch=2)
