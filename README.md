# Interpreting Generalized Category Discovery: Analyzing Core Features and Limitations in CUB and Herbarium Categorization

The repository is a fork, building on top of the Parametric Classification for Generalized Category Discovery: A Baseline Study 

The drastic increase in available data has driven the need for advanced machine learning techniques capable of extracting meaningful insights when dealing with unlabeled data. This project presents and interprets a Generalized Category Discovery (GCD) method pSimGCD which is an enhanced version of the SimGCD method that categorizes data into observed and newly created categories. The pSimGCD approach is designed to allow easier interpretation with its positive activations. The pSimGCD method improves the performance of the CUB-200-2011 (CUB) bird dataset by 0.76\%. Additionally, it demonstrates that when interpreted by CRAFT or the last self-attention layer, the model focuses mostly on key distinctive features for both the CUB and Herbarium datasets, implying it focuses on features that are also considered by humans. It also presents that the herbarium results are interpreted differently by the models and people because models identify herbarium categories on multiple features, unlike humans. It also suggests that image backgrounds affect the predictions as the habitats or the white background is included as a core interpretation feature in several categories.

Training a model: `bash scripts/run_${DATASET_NAME}.sh`

Generating the most important concepts generated by CRAFT: `bash scripts/run_craft.sh`

Generating images with attention: `bash scripts/run_attention.sh`

## Results
- pSimGCD accuracy performance compared to SimGCD on the CUB dataset
  | CUB | All | Old | New |
  |----------|----------|----------|----------|
  | SimGCD   | 61.5 ± 0.5   | 65.7 ± 0.5   | 59.4 ± 0.8   |
  | pSimGCD   | 62.28 ± 0.54   | 68.02 ± 1.0   | 59.4 ± 0.41   |

- Attention
  - ![attention_cub_last_23](https://github.com/user-attachments/assets/3b423772-a3b8-467f-a5ed-93b3984dc5d0)
  - ![attention_cub_last_5](https://github.com/user-attachments/assets/c3dc4a8e-41dc-41b5-86bc-52ad46893cb8)
  - ![attention_cub_last_13](https://github.com/user-attachments/assets/c7b17c7b-75c4-4eb5-bc4f-220546286959)
- CRAFT
  ![patch_size_legend_12](https://github.com/user-attachments/assets/4df23ca3-ac68-4e41-a281-376e2eab5bba)
