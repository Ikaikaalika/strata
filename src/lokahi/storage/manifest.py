"""Build runtime-neutral model specs from decoder-layer manifests."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..core.model_spec import ModelSpec, WeightGroup
from ..core.tensor_ref import TensorRef


def decoder_model_spec(
    model_id: str,
    layer_manifests: Iterable[Mapping[str, str]],
    loader: Any,
) -> ModelSpec:
    """Create one weight group per decoder layer.

    ``layer_manifests`` maps logical attribute names to storage keys. Loaders
    must expose ``tensor_metadata(storage_key)`` without loading tensor data.
    """
    metadata_reader = getattr(loader, "tensor_metadata", None)
    if metadata_reader is None:
        raise TypeError("loader must define tensor_metadata(name)")

    groups = []
    for layer_index, manifest in enumerate(layer_manifests):
        refs = []
        for logical_name, storage_key in manifest.items():
            metadata = metadata_reader(storage_key)
            refs.append(
                TensorRef(
                    name=logical_name,
                    storage_key=storage_key,
                    shape=tuple(int(value) for value in metadata["shape"]),
                    dtype=str(metadata["dtype"]),
                    storage_nbytes=(
                        None
                        if metadata.get("nbytes") is None
                        else int(metadata["nbytes"])
                    ),
                )
            )
        groups.append(
            WeightGroup(
                group_id=f"decoder.layer.{layer_index}",
                order=layer_index,
                tensors=tuple(refs),
            )
        )

    return ModelSpec.from_groups(model_id=model_id, groups=groups)
