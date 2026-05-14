from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="nvidia", device_name="cuda", device_query_cmd="nvidia-smi"
)
ARCH_MAP = {"9": "hopper", "8": "ampere"}


__all__ = ["*"]
