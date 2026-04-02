# Open set Spatio-temporal Phenological Segmentation

This project implements open-set spatio-temporal phenological segmentation using deep learning models.

## Dependencies

- [spatio-temporal-phenological-segmentation](https://github.com/keillernogueira/spatio-temporal-phenological-segmentation/)
- [OpenPCS](https://github.com/hugo-oliveira/openseg)
- [OpenPCS++](https://github.com/DiMorten/FCN_ConvLSTM_Crop_Recognition_Open_Set/blob/coords3/networks/convlstm_networks/train_src/analysis/open_set.py)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the implementation:
```bash
python test_implementation.py
```

3. Prepare your dataset with:
   - Time-series TIFF images (.tif files)
   - A mask file named `mask_train_test_int.png`

## Usage

### Training
```bash
python main.py --operation train --output_path ./output --dataset_path ./data --images img1 img2 img3 --patch_size 25 --hidden_class 2 --network fcn
```

### Testing
```bash
python main.py --operation test --output_path ./output --dataset_path ./data --images img1 img2 img3 --patch_size 25 --hidden_class 2 --network fcn --model_path ./output/best_model.pth
```

### Open-set Segmentation
```bash
python main.py --operation openset --output_path ./output --dataset_path ./data --images img1 img2 img3 --patch_size 25 --hidden_class 2 --network fcn --model_path ./output/best_model.pth
```

## Arguments

- `--operation`: Operation to perform (train/test/openset)
- `--output_path`: Path to save results
- `--dataset_path`: Path to dataset
- `--images`: List of image names (without .tif extension)
- `--patch_size`: Patch size for processing
- `--hidden_class`: Hidden class for open-set (0-3)
- `--network`: Network type (grsl/fcn)
- `--model_path`: Path to trained model (required for test/openset)