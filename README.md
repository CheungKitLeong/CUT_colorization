This repo is a course project of AIST-4010 in CUHK.

# Unpaired Style-specific Colorization with Contrastive Learning

The details of the project is presented in [report.pdf](report.pdf).

For the usage of the original cut framework, please check [README_CUT.md](README_CUT.md)

## Datasets

The preprocessed dataset can be download form [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155143469_link_cuhk_edu_hk/ET2hr57Sf1hLvmwGLUaVNT4BetwgJrWcu8sQ9D9NTkSvNw?e=Yk2fKS)
Please unzip the file and move the directory to `datasets\`

## Usage

Call [the project model](models/project_model.py) by the following command

### Training
```bash
python train.py --dataroot ./datasets/photo2vangogh --model project --name project_demo
```
The following options can control the weight of additional loss funcitons
```bash
--lambda_TV
--lambda_identity
--lambda_color
```
### Testing
```bash
python test.py --dataroot ./datasets/photo2vangogh --model project name --p2v_color_var
```
There are 4 variants of the pretrained model, refer to [checkpoints](checkpoint/)
