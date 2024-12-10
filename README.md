# LingTea

Source code for our *Findings of EMNLP 2024* paper [Cross-Lingual Unlearning of Selective Knowledge in Multilingual Language Models](https://aclanthology.org/2024.findings-emnlp.630/).

## Requirements

- Python (tested on 3.11.9)
- CUDA (tested on 12.1)
- PyTorch (tested on 2.3.1)
- PyTorch Lightning (tested on 2.3.1)
- Transformers (tested on 4.44.2)
- Datasets (tested on 2.20.0)
- torchmetrics (tested on 1.4.0)
- wandb (tested on 0.17.5)

## Unlearning

Unlearn a subset of FLORES-200:

```shell
>> bash scripts/run_flores.sh
```

Unlearn a subset of BMLAMA-53:

```shell
>> bash scripts/run_bmlama.sh
```

## Citation

If you make use of this code in your work, please kindly cite our paper:

```bibtex
@inproceedings{choi2024lingtea,
    title = "Cross-Lingual Unlearning of Selective Knowledge in Multilingual Language Models",
    author = "Choi, Minseok  and
      Min, Kyunghyun  and
      Choo, Jaegul",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.630",
    doi = "10.18653/v1/2024.findings-emnlp.630",
    pages = "10732--10747",
    abstract = "Pretrained language models memorize vast amounts of information, including private and copyrighted data, raising significant safety concerns. Retraining these models after excluding sensitive data is prohibitively expensive, making machine unlearning a viable, cost-effective alternative. Previous research has focused on machine unlearning for monolingual models, but we find that unlearning in one language does not necessarily transfer to others. This vulnerability makes models susceptible to low-resource language attacks, where sensitive information remains accessible in less dominant languages. This paper presents a pioneering approach to machine unlearning for multilingual language models, selectively erasing information across different languages while maintaining overall performance. Specifically, our method employs an adaptive unlearning scheme that assigns language-dependent weights to address different language performances of multilingual language models. Empirical results demonstrate the effectiveness of our framework compared to existing unlearning baselines, setting a new standard for secure and adaptable multilingual language models.",
}
```
