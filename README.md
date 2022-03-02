# Aura

## Setup

Create a conda environment from the requirement.yml file

```
conda env create --file requirement.yml
```

## Running the Aura Sampler
### 1. Modify the following fields in ```configs/main_config.yml```

**data_csv**: Location of csv containing paths to data files

**n**: Number of samples you want (default: 1000)

**sampling_method**: Whether you want to sample by "rank" or "diversity" (default:"diversity")

**column_name**: Name of the column in the csv with filenames (default:file_url.csv)

### 2. Run the following script

```
main.py --config ../configs/main_config.yml --clean_speech_classifier ../models/classifiers/labeler.onnx --sig_dnsmos_path ../models/dns835/sig.onnx --bak_ovr_dnsmos_path ../models/dns835/bak_ovr.onnx --noise_type_classifier_path ../models/noise_type_model/tagger_236.onnx --centroids ../models/models/centroids/centroids.npz --ood_centroids ../models/centroids/centroids_dns_train.npz --save_dir ./outputs
```

## Citation
If you use our work, please cite using the following bibtex:
```
@misc{gitiaux2021aura,
      title={Aura: Privacy-preserving augmentation to improve test set diversity in noise suppression applications}, 
      author={Xavier Gitiaux and Aditya Khant and Chandan Reddy and Jayant Gupchup and Ross Cutler},
      year={2021},
      eprint={2110.04391},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

## Link to paper:
[Aura: Privacy-preserving augmentation to improve test set diversity in noise suppression applications](https://arxiv.org/abs/2110.04391)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
