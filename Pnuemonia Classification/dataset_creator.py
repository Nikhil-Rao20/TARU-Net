import time
import shutil
from pathlib import Path

def combine_datasets_outliers(ds1_root, ds2_root, output_root):
    ds1_root    = Path(ds1_root)
    ds2_root    = Path(ds2_root)
    output_root = Path(output_root)

    # 1) First dataset mappings and splits
    ds1_class_map = {
        "normal":  "NORMAL",
        "opacity": "PNEUMONIA"
    }
    splits = ["train", "test"]

    for split in splits:
        for src_cls, dst_cls in ds1_class_map.items():
            src_dir  = ds1_root / split / src_cls
            dst_dir  = output_root / split / dst_cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                print(f"  [warn] {src_dir} does not exist, skipping.")
                continue

            for img in src_dir.iterdir():
                if img.is_file():
                    shutil.copy2(img, dst_dir / img.name)

    # 2) Second dataset goes entirely into train split
    ds2_class_map = {
        "Normal":       "NORMAL",
        # "Tuberculosis": "PNEUMONIA"
    }
    train_root = output_root / "train"
    for src_cls, dst_cls in ds2_class_map.items():
        src_dir = ds2_root / src_cls
        dst_dir = train_root / dst_cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  [warn] {src_dir} does not exist, skipping.")
            continue

        for img in src_dir.iterdir():
            if img.is_file():
                shutil.copy2(img, dst_dir / img.name)



def combine_datasets(dataset_dirs, ds1_root, ds2_root, output_dir):
    splits = ['train', 'test']
    output_dir = Path(output_dir)

    for split in splits:
        for dataset in dataset_dirs:
            src_split = Path(dataset) / split
            if not src_split.exists():
                print(f"Warning: {src_split} does not exist, skipping.")
                continue

            for class_dir in src_split.iterdir():
                if not class_dir.is_dir():
                    continue

                # make sure destination class folder exists
                dest_class_dir = output_dir / split / class_dir.name
                dest_class_dir.mkdir(parents=True, exist_ok=True)

                # copy all files from source class to destination class
                for img_path in class_dir.iterdir():
                    if img_path.is_file():
                        # if you want to overwrite duplicates, use copy; otherwise use copy2 to preserve metadata
                        shutil.copy2(img_path, dest_class_dir / img_path.name)

    combine_datasets_outliers(ds1_root, ds2_root, output_dir)
    
if __name__ == "__main__":
    a = time.time()
    combine_datasets(dataset_dirs=["/kaggle/input/chest-xray-covid19-pneumonia/Data", "/kaggle/input/chest-x-ray-image/Data", "/kaggle/input/pediatric-pneumonia-chest-xray/Pediatric Chest X-ray Pneumonia", "/kaggle/input/labeled-chest-xray-images/chest_xray",'/kaggle/input/chest-xray-pneumonia/chest_xray'],ds1_root = "/kaggle/input/pneumonia-xray-images",ds2_root = "/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database", output_dir = "/kaggle/working/Combined_Dataset")
    print("Done! All datasets have been merged into", "Combined_Dataset")
    print("Time required is: ", time.time()-a)
