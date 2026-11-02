# Historical dataset lookup

This root exists only to resolve immutable problem IDs in previously exported
KS3 manifests. Set `OP_VERIFY_PROBLEMS_DIR` here when replaying those datasets;
keep their original KS3 prefix and let the manifest select the problem IDs.
The aliases load current shared adapters and vendor-dispatched references.

New discovery and exports use `verification/problems`, with the model-neutral
`operators` and `rope` groups. Do not use this root to create new datasets.
