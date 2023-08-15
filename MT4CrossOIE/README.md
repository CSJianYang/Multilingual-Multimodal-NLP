# MT4CrossOIE

![picture](https://www.yuweiyin.com/files/img/2023-08-15-MT4CrossOIE-Training.png)

## Abstract

Cross-lingual open information extraction aims
to extract structured information from raw text
across multiple languages. Previous work uses
a shared cross-lingual pre-trained model to
handle the different languages but underuses
the potential of the language-specific representation.
In this paper, we propose an effective
multi-stage tuning framework called
MT4CrossOIE, designed for enhancing cross-lingual
open information extraction by injecting
language-specific knowledge into the
shared model. Specifically, the cross-lingual
pre-trained model is first tuned in a shared
semantic space (e.g., embedding matrix) in
the fixed encoder and then other components
are optimized in the second stage. After
enough training, we freeze the pre-trained
model and tune the multiple extra low-rank
language-specific modules using mixture-of-
LoRAs for model-based cross-lingual transfer.
In addition, we leverage two-stage prompting
to encourage the large language model
(LLM) to annotate the multilingual raw data
for data-based cross-lingual transfer. The
model is trained with multilingual objectives
on our proposed dataset OpenIE4++ by combing
the model-based and data-based transfer
techniques. Experimental results on various
benchmarks emphasize the importance of aggregating
multiple plug-in-and-play languagespecific
modules and demonstrate the effectiveness
of MT4CrossOIE in cross-lingual OIE.


## Datasets

* [OpenIE4++ Dataset](https://drive.google.com/drive/folders/1OW8lzVBFfmAAVtLXWutSIq1RxGU9TgON?usp=sharing)


## Architecture

![picture](https://www.yuweiyin.com/files/img/2023-08-15-MT4CrossOIE-Model.png)


## License

Please refer to the [LICENSE](./LICENSE) file for more details.


## Citation

* arXiv: https://arxiv.org/abs/2308.06552

```bibtex
@article{wang2023mt4crossoie,
  title   = {MT4CrossOIE: Multi-stage Tuning for Cross-lingual Open Information Extraction},
  author  = {Wang, Zixiang and Chai, Linzheng and Yang, Jian and Bai, Jiaqi and Yin, Yuwei and 
             Liu, Jiaheng and Guo, Hongcheng and Li, Tongliang and Yang, Liqun and
             Hebboul, Zine el-abidine and Li, Zhoujun},
  journal = {arXiv preprint arXiv:2308.06552},
  year    = {2023},
}
```
