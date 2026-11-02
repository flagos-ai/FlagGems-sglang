# Historical dataset lookup

This root exists only to resolve immutable problem IDs in previously exported
KS3 manifests. Set `OP_VERIFY_PROBLEMS_DIR` here when replaying those datasets;
keep their original KS3 prefix and let the manifest select the problem IDs.
The `qwen36`, `operators` and `rope` aliases load current `_plugin` adapters
and vendor-dispatched references. Both the original and model-neutral object
IDs resolve without renaming or uploading a tensor.

New discovery and exports use `verification/problems`, with the model-neutral
`operators` and `rope` groups. Do not use this root to create new datasets.

Use `python verification/check_plugins.py --list` to inspect the explicit map,
or `python verification/check_plugins.py --device <device>` to replay all eight
plugin names against their original published NVIDIA datasets. JSON results use
plugin names and retain `dataset_problem`/`dataset_prefix` provenance. The runner
only reads existing KS3 objects; exports and uploads are not part of replay.
