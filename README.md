# Rat liver microsomal stability

Flags compounds likely to be cleared quickly by rat liver microsomes, an early species read on whether exposure can be held long enough for efficacy work, with a half-life under 30 minutes counting as unstable. Siramshetty and colleagues at NCATS trained a graph convolutional network on roughly 25,000 in-house Tier I measurements and then, unusually, tested it on marketed drugs rather than only a held-out split, where balanced accuracy dropped to 0.71. Rat microsomal turnover remains an imperfect surrogate for human metabolism.

This model was incorporated on 2023-01-02.Last packaged on 2025-10-15.

## Information
### Identifiers
- **Ersilia Identifier:** `eos5505`
- **Slug:** `ncats-rlm`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `ADMET`
- **Target Organism:** `Rattus norvegicus`
- **Tags:** `Microsomal stability`, `Rat`, `ADME`, `Metabolism`, `Half-life`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability that a compound is unstable in rat liver microsomes, with instability set at a half-life under 30 minutes.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| rlm_proba1 | float | high | Probability of the compound being metabolised by rat liver microsomes at half-life below 30 min |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos5505](https://hub.docker.com/r/ersiliaos/eos5505)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5505.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5505.zip)

### Resource Consumption
- **Model Size (Mb):** `86`
- **Environment Size (Mb):** `2461`
- **Image Size (Mb):** `2592.73`

**Computational Performance (seconds):**
- 10 inputs: `28.93`
- 100 inputs: `18.78`
- 10000 inputs: `114.73`

### References
- **Source Code**: [https://github.com/ncats/ncats-adme](https://github.com/ncats/ncats-adme)
- **Publication**: [https://doi.org/10.1177/24725552211017520](https://doi.org/10.1177/24725552211017520)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2021`
- **Ersilia Contributor:** [pauline-banye](https://github.com/pauline-banye)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [None](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos5505
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos5505
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
