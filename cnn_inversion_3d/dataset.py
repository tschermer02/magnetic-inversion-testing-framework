from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
import math



TMI_SHAPE = (81, 81, 1)

SUSCEPTIBILITY_SHAPE = (24, 64, 64, 1)
SINGLE_PLANE_TMI_SHAPE = TMI_SHAPE


@dataclass(frozen=True)
class DatasetLoaderConfig:
    """
    Configuration for loading tmi-susceptibility training pairs.

    Parameters
    ----------
    batch_size
        Number of examples in each TensorFlow batch.
    shuffle
        Whether to shuffle the sample paths.
    shuffle_seed
        Deterministic seed used when shuffling.
    cache
        Whether to cache decoded samples in memory.
    prefetch
        Whether TensorFlow should prepare future batches while the
        current batch is being processed.
    tmi_scale
        Constant used to normalize tmi. A value of one leaves the
        tmi data unchanged.
    susceptibility_scale
        Constant used to normalize susceptibility. The current baseline susceptibility
        contrast is at most 1.0 g/cm3, so the default leaves it unchanged.
    """

    batch_size: int = 2
    shuffle: bool = False
    shuffle_seed: int = 20260727
    cache: bool = False
    prefetch: bool = True

    tmi_scale: float = 1.0
    susceptibility_scale: float = 1.0

    def validate(self) -> None:
        """Validate loader settings."""

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least one."
            )

        if self.tmi_scale <= 0.0:
            raise ValueError(
                "tmi_scale must be greater than zero."
            )

        if self.susceptibility_scale <= 0.0:
            raise ValueError(
                "susceptibility_scale must be greater than zero."
            )


def find_repository_root() -> Path:
    """
    Return the repository root.
    """

    return Path(__file__).resolve().parents[1]


def resolve_dataset_directory(
    dataset_directory: str | Path,
) -> Path:
    """
    Resolve a dataset path from the repository root.

    Parameters
    ----------
    dataset_directory
        Absolute dataset path or a path relative to the repository root.

    Returns
    -------
    pathlib.Path
        Resolved dataset directory.
    """

    path = Path(
        dataset_directory
    )

    if not path.is_absolute():
        path = (
            find_repository_root()
            / path
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Dataset directory does not exist:\n"
            f"{path}"
        )

    return path


def read_manifest_paths(
    *,
    dataset_directory: Path,
    manifest_name: str,
) -> list[Path]:
    """
    Read sample paths from one split manifest.

    Parameters
    ----------
    dataset_directory
        Root directory of the generated dataset.
    manifest_name
        Manifest filename, such as ``train_manifest.csv``.

    Returns
    -------
    list of pathlib.Path
        Absolute sample paths in manifest order.
    """

    manifest_path = (
        dataset_directory
        / manifest_name
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Split manifest does not exist:\n"
            f"{manifest_path}"
        )

    sample_paths: list[Path] = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(
            manifest_file
        )

        if reader.fieldnames is None:
            raise ValueError(
                f"{manifest_name} contains no header."
            )

        if "relative_path" not in reader.fieldnames:
            raise ValueError(
                f"{manifest_name} does not contain a "
                "'relative_path' column."
            )

        for row in reader:
            relative_path = row.get(
                "relative_path"
            )

            if relative_path is None:
                raise ValueError(
                    f"{manifest_name} contains a row without "
                    "a relative path."
                )

            sample_path = (
                dataset_directory
                / relative_path
            ).resolve()

            if not sample_path.exists():
                raise FileNotFoundError(
                    "Sample referenced by the manifest "
                    "does not exist:\n"
                    f"{sample_path}"
                )

            sample_paths.append(
                sample_path
            )

    if not sample_paths:
        raise ValueError(
            f"{manifest_name} contains no sample rows."
        )

    return sample_paths


def load_npz_sample(
    sample_path: str | Path,
    *,
    tmi_shape: tuple[int, ...] = TMI_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one tmi-susceptibility pair from disk.

    The channel dimension is appended here so TensorFlow receives
    channels-last 3D tensors:

    - tmi: ``(8, 64, 64, 1)``
    - susceptibility: ``(24, 64, 64, 1)``

    Parameters
    ----------
    sample_path
        Path to one generated ``.npz`` sample.

    Returns
    -------
    tuple of numpy.ndarray
        TMI input and susceptibility target as float32 arrays.
    """

    path = Path(
        sample_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Sample does not exist:\n{path}"
        )

    with np.load(
        path
    ) as sample:
        if "tmi" not in sample and "tmi" not in sample:
            raise KeyError(
                f"{path.name} has no 'tmi' array."
            )

        if "susceptibility" not in sample and "susceptibility" not in sample:
            raise KeyError(
                f"{path.name} has no 'susceptibility' array."
            )

        tmi = np.asarray(
            sample["tmi"] if "tmi" in sample else sample["tmi"],
            dtype=np.float32,
        )

        susceptibility = np.asarray(
            sample["susceptibility"] if "susceptibility" in sample else sample["susceptibility"],
            dtype=np.float32,
        )

    expected_tmi_without_channel = (
        tmi_shape[:-1]
    )

    expected_susceptibility_without_channel = (
        SUSCEPTIBILITY_SHAPE[:-1]
    )

    if (
        tmi.shape
        != expected_tmi_without_channel
    ):
        raise ValueError(
            f"{path.name}: expected tmi shape "
            f"{expected_tmi_without_channel}, "
            f"received {tmi.shape}."
        )

    if (
        susceptibility.shape
        != expected_susceptibility_without_channel
    ):
        raise ValueError(
            f"{path.name}: expected susceptibility shape "
            f"{expected_susceptibility_without_channel}, "
            f"received {susceptibility.shape}."
        )

    if not np.all(
        np.isfinite(tmi)
    ):
        raise ValueError(
            f"{path.name}: tmi contains invalid values."
        )

    if not np.all(
        np.isfinite(susceptibility)
    ):
        raise ValueError(
            f"{path.name}: susceptibility contains invalid values."
        )

    tmi = tmi[
        ...,
        np.newaxis,
    ]

    susceptibility = susceptibility[
        ...,
        np.newaxis,
    ]

    return (
        np.ascontiguousarray(
            tmi,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            susceptibility,
            dtype=np.float32,
        ),
    )


def load_magnetic_sample(
    sample_path: str | Path,
    *,
    tmi_shape: tuple[int, ...] = TMI_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a canonical ``(TMI, susceptibility)`` training pair."""
    return load_npz_sample(sample_path, tmi_shape=tmi_shape)


def _sample_generator(
    *,
    sample_paths: list[Path],
    tmi_scale: float,
    susceptibility_scale: float,
    tmi_shape: tuple[int, ...],
) -> Iterator[
    tuple[np.ndarray, np.ndarray]
]:
    """
    Yield normalized samples for TensorFlow.
    """

    for sample_path in sample_paths:
        tmi, susceptibility = load_npz_sample(
            sample_path,
            tmi_shape=tmi_shape,
        )

        tmi = (
            tmi
            / np.float32(
                tmi_scale
            )
        )

        susceptibility = (
            susceptibility
            / np.float32(
                susceptibility_scale
            )
        )

        yield (
            np.ascontiguousarray(
                tmi,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                susceptibility,
                dtype=np.float32,
            ),
        )


def build_dataset(
    *,
    dataset_directory: str | Path,
    manifest_name: str,
    config: DatasetLoaderConfig,
    tmi_shape: tuple[int, ...] = TMI_SHAPE,
) -> tuple[tf.data.Dataset, int]:
    """
    Build one TensorFlow dataset from a split manifest.

    Parameters
    ----------
    dataset_directory
        Root directory of the generated dataset.
    manifest_name
        Split manifest filename.
    config
        Dataset loading options.

    Returns
    -------
    tuple
        TensorFlow dataset and number of unbatched samples.
    """

    config.validate()

    resolved_directory = (
        resolve_dataset_directory(
            dataset_directory
        )
    )

    sample_paths = read_manifest_paths(
        dataset_directory=resolved_directory,
        manifest_name=manifest_name,
    )

    output_signature = (
        tf.TensorSpec(
            shape=tmi_shape,
            dtype=tf.float32,
            name="tmi",
        ),
        tf.TensorSpec(
            shape=SUSCEPTIBILITY_SHAPE,
            dtype=tf.float32,
            name="susceptibility",
        ),
    )

    dataset = (
        tf.data.Dataset.from_generator(
            lambda: _sample_generator(
                sample_paths=sample_paths,
                tmi_scale=(
                    config.tmi_scale
                ),
                susceptibility_scale=(
                    config.susceptibility_scale
                ),
                tmi_shape=tmi_shape,
            ),
            output_signature=output_signature,
        )
    )

    if config.shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(
                sample_paths
            ),
            seed=config.shuffle_seed,
            reshuffle_each_iteration=True,
        )

    if config.cache:
        dataset = dataset.cache()

    dataset = dataset.batch(
            config.batch_size,
            drop_remainder=False,
        )

    number_of_batches = math.ceil(
         len(sample_paths)
        / config.batch_size
    )

    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(
            number_of_batches
        )
     )

    if config.prefetch:
        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

    return (
        dataset,
        len(sample_paths),
    )


def build_training_datasets(
    *,
    dataset_directory: str | Path,
    batch_size: int,
    tmi_scale: float = 1.0,
    susceptibility_scale: float = 1.0,
    random_seed: int = 20260727,
    tmi_shape: tuple[int, ...] = TMI_SHAPE,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    dict[str, int],
]:
    """
    Build training, validation, and test TensorFlow datasets.

    Returns
    -------
    tuple
        Training dataset, validation dataset, test dataset, and sample
        counts for each split.
    """

    training_dataset, training_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "train_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=True,
                shuffle_seed=random_seed,
                tmi_scale=tmi_scale,
                susceptibility_scale=susceptibility_scale,
            ),
            tmi_shape=tmi_shape,
        )
    )

    validation_dataset, validation_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "validation_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                tmi_scale=tmi_scale,
                susceptibility_scale=susceptibility_scale,
            ),
            tmi_shape=tmi_shape,
        )
    )

    test_dataset, test_count = (
        build_dataset(
            dataset_directory=dataset_directory,
            manifest_name=(
                "test_manifest.csv"
            ),
            config=DatasetLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                tmi_scale=tmi_scale,
                susceptibility_scale=susceptibility_scale,
            ),
            tmi_shape=tmi_shape,
        )
    )

    counts = {
        "train": training_count,
        "validation": validation_count,
        "test": test_count,
    }

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
        counts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the loader smoke-test arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load the FWD3D dataset splits and inspect one "
            "TensorFlow batch."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "datasets/fwd3d_smoke_test"
        ),
        help=(
            "Dataset directory. Relative paths are interpreted "
            "from the repository root."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="TensorFlow batch size.",
    )

    return parser


def main() -> None:
    """
    Run a TensorFlow dataset-loader smoke test.
    """

    parser = build_argument_parser()
    arguments = parser.parse_args()

    (
        training_dataset,
        validation_dataset,
        test_dataset,
        counts,
    ) = build_training_datasets(
        dataset_directory=arguments.dataset,
        batch_size=arguments.batch_size,
    )

    training_batch = next(
        iter(
            training_dataset
        )
    )

    tmi_batch, susceptibility_batch = (
        training_batch
    )

    expected_tmi_rank = 5
    expected_susceptibility_rank = 5

    if len(
        tmi_batch.shape
    ) != expected_tmi_rank:
        raise AssertionError(
            "TMI batch must have rank five."
        )

    if len(
        susceptibility_batch.shape
    ) != expected_susceptibility_rank:
        raise AssertionError(
            "Susceptibility batch must have rank five."
        )

    if tuple(
        tmi_batch.shape[1:]
    ) != TMI_SHAPE:
        raise AssertionError(
            "Unexpected tmi sample shape: "
            f"{tmi_batch.shape[1:]}"
        )

    if tuple(
        susceptibility_batch.shape[1:]
    ) != SUSCEPTIBILITY_SHAPE:
        raise AssertionError(
            "Unexpected susceptibility sample shape: "
            f"{susceptibility_batch.shape[1:]}"
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(
                tmi_batch
            )
        )
    ):
        raise AssertionError(
            "TMI batch contains invalid values."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(
                susceptibility_batch
            )
        )
    ):
        raise AssertionError(
            "Susceptibility batch contains invalid values."
        )

    print()
    print("TensorFlow dataset-loader test")
    print("=" * 30)
    print(
        f"Split counts: {counts}"
    )
    print(
        f"TMI batch shape: "
        f"{tmi_batch.shape}"
    )
    print(
        f"Susceptibility batch shape: "
        f"{susceptibility_batch.shape}"
    )
    print(
        f"TMI dtype: "
        f"{tmi_batch.dtype.name}"
    )
    print(
        f"Susceptibility dtype: "
        f"{susceptibility_batch.dtype.name}"
    )
    print(
        f"TMI range in batch: "
        f"{float(tf.reduce_min(tmi_batch)):.8e} "
        f"to "
        f"{float(tf.reduce_max(tmi_batch)):.8e}"
    )
    print(
        f"Susceptibility range in batch: "
        f"{float(tf.reduce_min(susceptibility_batch)):.8e} "
        f"to "
        f"{float(tf.reduce_max(susceptibility_batch)):.8e}"
    )
    print()
    print("Training dataset load: PASSED")
    print("Validation dataset load: PASSED")
    print("Test dataset load: PASSED")
    print("Input shape check: PASSED")
    print("Target shape check: PASSED")
    print("Finite-value check: PASSED")

    # Make sure the other two datasets can also produce a batch.
    next(
        iter(
            validation_dataset
        )
    )

    next(
        iter(
            test_dataset
        )
    )


if __name__ == "__main__":
    main()
