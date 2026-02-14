# WAV Fast-Path Experiment

Date: 2026-02-14

Added a native WAV parser (PCM + IEEE float) to avoid ffmpeg subprocess startup for supported `.wav` files.

| Scenario | Before Mean (s) | After Mean (s) | Delta |
|---|---:|---:|---:|
| fp16 short | 0.5450552458045422 | 0.5056636458333136 | -7.227083910197707% |
| fp16 long 10s | 0.9125695416278177 | 0.9162171980005951 | 0.39971270203373876% |
| 4bit-g64 short | 0.15749083319969942 | 0.11846737483332011 | -24.778241103656683% |
| 4bit-g64 long 10s | 0.24907784337301564 | 0.22578043224893918 | -9.353465891860392% |

Stage-level profiling on `tests/fixtures/test_speech.wav` also shows `load_audio` time dropping from ~90ms to sub-1ms when the fast path is used.

