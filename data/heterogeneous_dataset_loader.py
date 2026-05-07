import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel is not installed, .nii loading is unavailable")
    print("Install with: pip install nibabel")


class HeterogeneousBraTSDataset(Dataset):
    """Dataset for text-only, image-only, and multimodal federated clients."""

    def __init__(
        self,
        data_dir: str,
        mode: str = "private",
        client_type: str = "multimodal",
        image_size: int = 256,
        max_samples: Optional[int] = None,
        load_mask: bool = True,
        include_text_features: bool = True,
        is_validation: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.client_type = client_type
        self.image_size = image_size
        self.load_mask = load_mask
        self.include_text_features = include_text_features
        self.is_validation = is_validation
        self.samples: Optional[List[Tuple[int, int]]] = None
        self._all_zero_mask_warned_cases: set[str] = set()

        self.case_dir = self.data_dir / ("private" if mode == "private" else "public")
        if not self.case_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.case_dir}")

        if self.client_type == "text_only":
            text_files = sorted(self.case_dir.glob("**/*_text.npy"))
            if not text_files:
                raise ValueError(
                    f"[HeterogeneousBraTSDataset text_only] No *_text.npy files found in {self.case_dir}"
                )
            if max_samples is not None and max_samples > 0:
                text_files = text_files[:max_samples]
            self.text_files = text_files
            self.case_folders: List[Path] = []
            print(
                f"  [HeterogeneousDataset] {mode.capitalize()} - {client_type} - loaded {len(self.text_files)} text samples"
            )
        else:
            self.case_folders = sorted(
                [folder for folder in self.case_dir.iterdir() if folder.is_dir() and "BraTS" in folder.name]
            )
            if not self.case_folders:
                raise ValueError(f"No BraTS case folders found in {self.case_dir}")

            if max_samples is not None and max_samples > 0:
                self.case_folders = self.case_folders[:max_samples]

            self.text_files = []
            if self.is_validation and self.load_mask:
                self.samples = self._build_validation_samples()
                print(
                    f"  [HeterogeneousDataset] {mode.capitalize()} - {client_type} - loaded {len(self.case_folders)} cases / {len(self.samples)} validation slices"
                )
            else:
                print(
                    f"  [HeterogeneousDataset] {mode.capitalize()} - {client_type} - loaded {len(self.case_folders)} cases"
                )

    def __len__(self) -> int:
        if self.client_type == "text_only":
            return len(self.text_files)
        if self.samples is not None:
            return len(self.samples)
        return len(self.case_folders)

    def __getitem__(self, idx: int):
        """
        Returns:
        - text_only: (text_feature,)
        - image_only: (image, mask) or (image,)
        - multimodal:
            - training private: (image, mask, text_feature)
            - training public:  (image, text_feature)
            - validation:       (image, mask)
        """
        if self.client_type == "text_only":
            text_path = self.text_files[idx]
            try:
                text_feature = np.load(str(text_path))
                return (torch.from_numpy(text_feature).float(),)
            except Exception as exc:
                raise RuntimeError(
                    f"[HeterogeneousBraTSDataset] Failed to load text feature: {text_path}\nReason: {exc}"
                ) from exc

        slice_idx = None
        if self.samples is not None:
            case_idx, slice_idx = self.samples[idx]
            case_folder = self.case_folders[case_idx]
        else:
            case_folder = self.case_folders[idx]
            if self.client_type in {"image_only", "multimodal"} and self.load_mask:
                slice_idx = self._select_slice_stratified(case_folder)

        text_feature = None
        if self.client_type == "multimodal" and self.include_text_features:
            text_feature = self._load_text_feature(case_folder)

        image = None
        if self.client_type in {"image_only", "multimodal"}:
            image = self._load_image(case_folder, slice_idx=slice_idx)

        mask = None
        if self.load_mask and self.client_type in {"image_only", "multimodal"}:
            mask = self._load_mask(case_folder, slice_idx=slice_idx)

        if self.client_type == "image_only":
            if mask is not None:
                return image, mask
            return (image,)

        if self.client_type == "multimodal":
            if mask is not None and text_feature is not None:
                return image, mask, text_feature
            if mask is not None:
                return image, mask
            if text_feature is not None:
                return image, text_feature
            return (image,)

        raise ValueError(f"Unknown client_type: {self.client_type}")

    def _build_validation_samples(self) -> List[Tuple[int, int]]:
        """Build deterministic (case_idx, slice_idx) pairs for validation."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required for validation slice enumeration")

        samples: List[Tuple[int, int]] = []
        for case_idx, case_folder in enumerate(self.case_folders):
            mask_file = self._find_first_file(case_folder, ["*_seg.nii", "*_seg.nii.gz"])
            if mask_file is None:
                continue

            mask_data = nib.load(mask_file).get_fdata()
            if mask_data.ndim != 3:
                continue

            for slice_idx in range(mask_data.shape[2]):
                samples.append((case_idx, slice_idx))

        if not samples:
            warnings.warn(
                f"[HeterogeneousBraTSDataset] No validation slices built from {self.case_dir}"
            )
        return samples

    def _find_first_file(self, case_folder: Path, patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            matches = sorted(case_folder.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def _load_text_feature(self, case_folder: Path) -> torch.Tensor:
        text_files = sorted(case_folder.glob("*_text.npy"))
        if not text_files:
            raise FileNotFoundError(f"No text feature file found in {case_folder}")

        text_file = text_files[0]
        for candidate in text_files:
            if "flair" in candidate.name:
                text_file = candidate
                break

        return torch.from_numpy(np.load(text_file)).float()

    def _select_slice_stratified(self, case_folder: Path) -> Optional[int]:
        mask_file = self._find_first_file(case_folder, ["*_seg.nii", "*_seg.nii.gz"])
        if mask_file is None:
            return None

        mask_data = nib.load(mask_file).get_fdata()
        if mask_data.ndim != 3:
            return None

        per_slice_fg = torch.from_numpy((mask_data > 0).sum(axis=(0, 1)).astype(np.int64))
        valid_indices = torch.where(per_slice_fg > 10)[0]
        if len(valid_indices) > 0:
            rand_pos = torch.randint(0, len(valid_indices), (1,)).item()
            return valid_indices[rand_pos].item()
        return torch.argmax(per_slice_fg).item()

    def _load_image(self, case_folder: Path, slice_idx: Optional[int] = None) -> torch.Tensor:
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required: pip install nibabel")

        image_file = self._find_first_file(
            case_folder,
            ["*_flair.nii", "*_flair.nii.gz", "*_t1.nii", "*_t1.nii.gz"],
        )
        if image_file is None:
            raise FileNotFoundError(f"No image file found in {case_folder}")

        image_data = nib.load(image_file).get_fdata()
        if image_data.ndim == 3:
            z = slice_idx if slice_idx is not None else image_data.shape[2] // 2
            z = max(0, min(z, image_data.shape[2] - 1))
            image_data = image_data[:, :, z]

        brain_mask = image_data > 0
        if brain_mask.sum() > 0:
            mu = image_data[brain_mask].mean()
            sigma = image_data[brain_mask].std()
            image_data = (image_data - mu) / (sigma + 1e-8)
            image_data[~brain_mask] = 0.0
        else:
            image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-8)

        image = torch.from_numpy(image_data).float()
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)

        return F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _load_mask(self, case_folder: Path, slice_idx: Optional[int] = None) -> torch.Tensor:
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required: pip install nibabel")

        mask_file = self._find_first_file(case_folder, ["*_seg.nii", "*_seg.nii.gz"])
        if mask_file is None:
            raise FileNotFoundError(f"No segmentation mask found in {case_folder}")

        mask_data = nib.load(mask_file).get_fdata()
        if mask_data.ndim == 3:
            if slice_idx is not None:
                z = max(0, min(slice_idx, mask_data.shape[2] - 1))
            else:
                per_slice_fg = torch.from_numpy((mask_data > 0).sum(axis=(0, 1)).astype(np.int64))
                z = torch.argmax(per_slice_fg).item()
            mask_data = mask_data[:, :, z]

        if mask_data.max() == 0 and case_folder.name not in self._all_zero_mask_warned_cases:
            self._all_zero_mask_warned_cases.add(case_folder.name)
            warnings.warn(
                f"[Dataset Warning] {case_folder.name}: selected slice is all-zero mask"
            )

        mask = torch.from_numpy(mask_data.copy()).long()
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        channel0 = (mask == 0).float()
        channel1 = (mask > 0).float()
        channel2 = (mask == 4).float()
        return torch.stack([channel0, channel1, channel2], dim=0)


def heterogeneous_collate_fn(batch, client_type: str):
    """Collate heterogeneous client batches into the expected tuple shapes."""
    if client_type == "text_only":
        text_features = [item[0] for item in batch]
        return (torch.stack(text_features, dim=0),)

    if client_type == "image_only":
        if len(batch[0]) == 2:
            images = [item[0] for item in batch]
            masks = [item[1] for item in batch]
            return torch.stack(images, dim=0), torch.stack(masks, dim=0)
        return (torch.stack([item[0] for item in batch], dim=0),)

    if client_type == "multimodal":
        if len(batch[0]) == 3:
            images = [item[0] for item in batch]
            masks = [item[1] for item in batch]
            text_features = [item[2] for item in batch]
            return (
                torch.stack(images, dim=0),
                torch.stack(masks, dim=0),
                torch.stack(text_features, dim=0),
            )
        if len(batch[0]) == 2:
            second = batch[0][1]
            images = [item[0] for item in batch]
            if isinstance(second, torch.Tensor) and second.dim() >= 2:
                masks = [item[1] for item in batch]
                return torch.stack(images, dim=0), torch.stack(masks, dim=0)
            text_features = [item[1] for item in batch]
            return torch.stack(images, dim=0), torch.stack(text_features, dim=0)
        return (torch.stack([item[0] for item in batch], dim=0),)

    raise ValueError(f"Unknown client_type: {client_type}")


def create_heterogeneous_data_loaders(
    data_root: str,
    split: str = "train",
    client_configs: List[Dict] = None,
    batch_size: int = 2,
    image_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    include_text_features: bool = True,
    is_validation: bool = False,
    load_public: Optional[bool] = None,
) -> Dict[str, Tuple[DataLoader, Optional[DataLoader]]]:
    """Create per-client private/public loaders for heterogeneous FL."""
    if client_configs is None:
        client_configs = [
            {"client_id": "client_1", "modality": "text_only"},
            {"client_id": "client_2", "modality": "image_only"},
            {"client_id": "client_3", "modality": "multimodal"},
        ]

    data_root_path = Path(data_root)
    loaders_dict = {}
    if load_public is None:
        load_public = split == "train"

    for config in client_configs:
        client_id = config["client_id"]
        modality = config["modality"]
        client_data_dir = data_root_path / split / client_id

        print(f"\n[{client_id}] Creating data loaders ({modality})...")

        try:
            private_dataset = HeterogeneousBraTSDataset(
                data_dir=str(client_data_dir),
                mode="private",
                client_type=modality,
                image_size=image_size,
                max_samples=max_samples,
                load_mask=True,
                include_text_features=include_text_features,
                is_validation=is_validation,
            )
            private_loader = DataLoader(
                private_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=lambda batch, m=modality: heterogeneous_collate_fn(batch, m),
                pin_memory=torch.cuda.is_available(),
            )
            print(f"  [OK] Private data: {len(private_dataset)} samples")
        except Exception as exc:
            print(f"  [ERROR] Failed to load private data: {exc}")
            raise

        public_loader = None
        if load_public:
            try:
                public_dataset = HeterogeneousBraTSDataset(
                    data_dir=str(client_data_dir),
                    mode="public",
                    client_type=modality,
                    image_size=image_size,
                    max_samples=max_samples,
                    load_mask=False,
                    include_text_features=include_text_features,
                    is_validation=False,
                )
                public_loader = DataLoader(
                    public_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=lambda batch, m=modality: heterogeneous_collate_fn(batch, m),
                    pin_memory=torch.cuda.is_available(),
                )
                print(f"  [OK] Public data: {len(public_dataset)} samples")
            except Exception as exc:
                print(f"  [WARN] Failed to load public data, skipping")
                print(f"    Reason: {exc}")

        loaders_dict[client_id] = (private_loader, public_loader)

    return loaders_dict


if __name__ == "__main__":
    print("heterogeneous_dataset_loader.py smoke entrypoint")
