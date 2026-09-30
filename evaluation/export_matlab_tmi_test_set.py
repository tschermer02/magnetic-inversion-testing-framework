"""Export E01 test susceptibility models to MATLAB-readable HDF5 files."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import h5py
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("datasets/E01_soft_tversky_full"))
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/matlab_python_tmi/inputs"))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.dataset/"test_manifest.csv").open(newline="",encoding="utf-8") as stream:rows=list(csv.DictReader(stream))
    exported=[]
    for index,row in enumerate(rows,1):
        with np.load(args.dataset/row["relative_path"]) as saved:
            susceptibility=np.asarray(saved["susceptibility"],np.float64)
            observed=np.asarray(saved["tmi"],np.float64)
        destination=args.output/f"{row['sample_id']}.h5"
        # Store one-dimensional vectors to make ordering explicit across languages.
        # MATLAB getPredMag expects X fastest, then Y, then Z. The receiver maps
        # are likewise flattened with X fastest.
        susceptibility_xfast = susceptibility.transpose(2, 1, 0).ravel(order="F")
        stored_tmi_xfast = observed.ravel(order="C")
        with h5py.File(destination, "w") as output:
            output.create_dataset("susceptibility_xfast_si", data=susceptibility_xfast)
            output.create_dataset("stored_tmi_xfast_nt", data=stored_tmi_xfast)
            output.attrs["sample_id"] = row["sample_id"]
        exported.append(row["sample_id"]);print(f"Exported {index}/{len(rows)}: {row['sample_id']}")
    metadata={"dataset":str(args.dataset.resolve()),"manifest":"test_manifest.csv","sample_count":len(exported),
        "source_model_order":"z,y,x","exported_model_order":"x-fastest, then y, then z",
        "exported_receiver_order":"x-fastest, then y","susceptibility_unit":"SI","tmi_unit":"nT",
        "grid":"64 x 64 x 24 cells; 10 m cubes; X east, Y north, Z down",
        "receivers":"81 x 81; X/Y 0..800 m by 10 m; Z=-10 m","sample_ids":exported}
    (args.output/"export_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"MATLAB inputs written to {args.output.resolve()}")
if __name__=="__main__":main()
