# Rat liver microsomal stability

Hepatic metabolic stability is key to ensure the drug attains the desired concentration in the body. The Rat Liver Microsomal (RLM) stability is a good approximation of a compound’s stability in the human body, and NCATS has collected a proprietary dataset of 20216 compounds with its associated RLM (in vitro half-life; unstable ≤30 min, stable >30 min) and used it to train a classifier based on an ensemble of several ML approaches (random forest, deep neural networks, graph convolutional neural networks and recurrent neural networks)

## Identifiers

* EOS model ID: `eos5505`
* Slug: `ncats-rlm`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Classification`
* Output: `Probability`
* Output Type: `Float`
* Output Shape: `Single`
* Interpretation: Probability of a compound being unstable in RLM assay (half-life ≤ 30min)

## References

* [Publication](https://slas-discovery.org/article/S2472-5552(22)06765-X/fulltext)
* [Source Code](https://github.com/ncats/ncats-adme)
* Ersilia contributor: [pauline-banye](https://github.com/pauline-banye)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos5505)
* [AWS S3](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5505.zip)
* [DockerHub](https://hub.docker.com/r/ersiliaos/eos5505) (AMD64, ARM64)

## Citation

If you use this model, please cite the [original authors](https://slas-discovery.org/article/S2472-5552(22)06765-X/fulltext) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a None license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!