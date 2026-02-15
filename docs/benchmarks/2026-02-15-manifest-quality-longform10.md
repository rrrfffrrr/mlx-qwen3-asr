# Manifest Quality (Long-Form 10x ~75-90s)

- samples: 10
- model: Qwen/Qwen3-ASR-0.6B (float16)
- wer: 0.1671
- cer: 0.0704
- primary error rate: 0.1156
- mean latency: 11.13s

Primary metric rule: CER for Chinese/Japanese/Korean; WER otherwise.

| language | samples | wer | cer | primary | latency (s) |
|---|---:|---:|---:|---:|---:|
| Arabic | 1 | 0.281 | 0.100 | 0.281 | 14.37 |
| Chinese | 1 | 0.476 | 0.018 | 0.018 | 6.55 |
| English | 1 | 0.080 | 0.029 | 0.080 | 8.65 |
| French | 1 | 0.139 | 0.060 | 0.139 | 11.81 |
| German | 1 | 0.061 | 0.022 | 0.061 | 12.01 |
| Hindi | 1 | 0.324 | 0.280 | 0.324 | 25.79 |
| Japanese | 1 | 0.895 | 0.109 | 0.109 | 7.22 |
| Korean | 1 | 0.103 | 0.040 | 0.040 | 7.55 |
| Russian | 1 | 0.152 | 0.047 | 0.152 | 9.52 |
| Spanish | 1 | 0.037 | 0.007 | 0.037 | 7.83 |
