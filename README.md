# Rat Liver Microsomonal Stability

## Model identifiers
- Slug: rlm_stability
- Ersilia ID: eos5505
- Tags: rlm
#
## Model description
<p align="justify">
Hepatic metabolic stability is a key parameter in drug discovery because it can prevent a drug from attaining sufficient in vivo exposure, producing short half-lives, poor oral bioavailability and low plasma concentrations. Metabolic stability is usually assessed in microsomal fractions and only the best compounds progress in the drug discovery process. Although a crucial endpoint, little or no data exists in the public domain. This data was provided by the National Center for Advancing Translational Sciences (NCATS).
</p>

- Input: SMILES
- Output: SMILES
<!-- - Model type: Classification -->
<!-- - Training set: (number of compounds and link to the training data)
- Mode of training: (is it pretrained? that is were the checkpoints downloaded and used to train the model? or is it retrained? that is trained from scratch with an updated data) -->
#
## Source code

Cite the source publication
[Retrospective assessment of rat liver microsomal stability at NCATS: data and QSAR models](https://pubmed.ncbi.nlm.nih.gov/33244000/)

- Code: [NCATS-ADME](https://github.com/ncats/ncats-adme.git)
- Checkpoints: include the link to the checkpoints used if model is a pretrained model
#
## License
GPL v3 license.
#
## History

- The model was incorporated into Ersilia on the 12th of January, 2023.
- Modifications to the original code.
    1. Removal of Flask functionalities and dependencies.
    2. Striping unused functions from the original code.
    3. Refactoring the original code to facilitate incorporation into the Ersilia Model Hub.

-   Steps to run the model on Ersilia CLI
    1. Install the [Ersilia Model Hub](https://ersilia.gitbook.io/ersilia-book/ersilia-model-hub/installation)
    2. Fetch, serve and run predictions on the model as outlined in the model usage section of the [Ersilia book](https://ersilia.gitbook.io/ersilia-book/ersilia-model-hub/antibiotic-activity-prediction)

#
## About us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission or [volunteer](https://www.ersilia.io/volunteer) with us!
