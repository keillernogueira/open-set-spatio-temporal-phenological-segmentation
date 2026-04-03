# run cergy
# train
CUDA_VISIBLE_DEVICES=0 python3.13 main.py --operation train --output_path output/hidden_3/ 
--dataset_path ../datasets/biologists_datasets/species_subset/ 
--images 2015_10_07_c 2015_11_13_c 2015_12_12_c 2016_02_23_c 2016_03_30_c 2016_04_26_c 2016_05_22_c 2016_06_20_c 2016_07_24_crop 2016_08_23_c 2016_09_25_c 2016_10_24_crop 2016_11_30_crop 2017_01_05_crop 2017_02_20_crop 
--patch_size 25 --hidden_class 3 --network fcn

# test
CUDA_VISIBLE_DEVICES=0 python3.13 main.py --operation test --output_path output/hidden_3/ 
--dataset_path ../datasets/biologists_datasets/species_subset/ 
--images 2015_10_07_c 2015_11_13_c 2015_12_12_c 2016_02_23_c 2016_03_30_c 2016_04_26_c 2016_05_22_c 2016_06_20_c 2016_07_24_crop 2016_08_23_c 2016_09_25_c 2016_10_24_crop 2016_11_30_crop 2017_01_05_crop 2017_02_20_crop 
--patch_size 25 --hidden_class 3 --network fcn --model_path output/hidden_3/model_11.pth

# openset
CUDA_VISIBLE_DEVICES=0 python3.13 main.py --operation openset --output_path output/hidden_3/ 
--dataset_path ../datasets/biologists_datasets/species_subset/ 
--images 2015_10_07_c 2015_11_13_c 2015_12_12_c 2016_02_23_c 2016_03_30_c 2016_04_26_c 2016_05_22_c 2016_06_20_c 2016_07_24_crop 2016_08_23_c 2016_09_25_c 2016_10_24_crop 2016_11_30_crop 2017_01_05_crop 2017_02_20_crop 
--patch_size 25 --hidden_class 3 --network fcn --model_path output/hidden_3/model_11.pth
