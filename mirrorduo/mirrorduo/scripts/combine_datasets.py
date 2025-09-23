import h5py
import os
from tqdm import tqdm
import argparse


def combine_hdf5(file_a, file_b, output_file, num_a=None, num_b=None):
    """
    Combine two HDF5 datasets into one, reindexing demos sequentially.

    Args:
        file_a (str): Path to first HDF5 file (source A).
        file_b (str): Path to second HDF5 file (source B).
        output_file (str): Path to save combined HDF5 output.
        num_a (int or None): Number of demos to take from A. If None, take all.
        num_b (int or None): Number of demos to take from B. If None, take all.
    """

    with h5py.File(file_a, "r") as f_a, h5py.File(file_b, "r") as f_b, h5py.File(
        output_file, "w"
    ) as f_out:
        data_grp = f_out.create_group("data")

        total_a = len(f_a["data"])
        total_b = len(f_b["data"])

        if num_a is None:
            num_a = total_a
        if num_b is None:
            num_b = total_b

        assert num_a <= total_a, f"Requested {num_a} demos from A, but only {total_a} available."
        assert num_b <= total_b, f"Requested {num_b} demos from B, but only {total_b} available."

        counter = 0

        # Copy demos from file A
        for i in tqdm(range(num_a), desc="Copying demos from A"):
            src_key = f"demo_{i}"
            dst_key = f"demo_{counter}"
            f_a.copy(f_a[f"data/{src_key}"], data_grp, name=dst_key)
            counter += 1

        # Copy demos from file B
        for i in tqdm(range(num_b), desc="Copying demos from B"):
            src_key = f"demo_{i}"
            dst_key = f"demo_{counter}"
            f_b.copy(f_b[f"data/{src_key}"], data_grp, name=dst_key)
            counter += 1

        # Save env_args from file A
        if "env_args" in f_a["data"].attrs:
            data_grp.attrs["env_args"] = f_a["data"].attrs["env_args"]
            print("Copied env_args from source A.")

        data_grp.attrs["total"] = counter

    print(f"Combined {num_a} demos from A + {num_b} demos from B into {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine two robomimic HDF5 datasets sequentially."
    )
    parser.add_argument(
        "--file_a", type=str, required=True, help="Path to first input HDF5 file (source A)."
    )
    parser.add_argument(
        "--file_b", type=str, required=True, help="Path to second input HDF5 file (source B)."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the combined output HDF5 file."
    )
    parser.add_argument(
        "--num_a", type=int, default=None, help="Number of demos to copy from file A."
    )
    parser.add_argument(
        "--num_b", type=int, default=None, help="Number of demos to copy from file B."
    )
    args = parser.parse_args()

    assert os.path.isfile(args.file_a), f"File {args.file_a} does not exist."
    assert os.path.isfile(args.file_b), f"File {args.file_b} does not exist."

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combine_hdf5(args.file_a, args.file_b, args.output, args.num_a, args.num_b)


if __name__ == "__main__":
    main()
