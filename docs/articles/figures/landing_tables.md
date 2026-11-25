
**Gemma 4 12B** (draft: l7bc, not exact, not counted)

| Context | Ours before | Ours after | Ours + draft | llama.cpp | llama.cpp + its draft | Our best / their best |
|---|---|---|---|---|---|---|
| 2k | 82 | 96 | 98 | 97 | 64 | 0.99× |
| 4k | 82 | 96 | 52 | 97 | 81 | 0.99× |
| 8k | 81 | 95 | 110 | 95 | 84 | 1.00× |
| 16k | 80 | 94 | 102 | 93 | 81 | 1.01× |
| 32k | 77 | 91 | 67 | 90 | 58 | 1.02× |
| 64k | – | 86 | 96 | 81 | 49 | 1.05× |
| 128k | – | 80 | – | 68 | 53 | 1.17× |

**Qwen3.8-27B** (draft: mtp4_h4, exact)

| Context | Ours before | Ours after | Ours + draft | llama.cpp | llama.cpp + its draft | Our best / their best |
|---|---|---|---|---|---|---|
| 2k | 57 | 73 | 176 | 76 | 158 | 1.11× |
| 4k | 56 | 72 | 136 | 74 | 127 | 1.07× |
| 8k | 55 | 69 | 110 | 73 | 117 | 0.94× |
| 16k | 52 | 65 | 133 | 70 | 127 | 1.05× |
| 32k | 47 | 58 | 97 | 61 | 90 | 1.08× |
| 64k | 40 | 47 | 83 | 49 | 87 | 0.95× |
| 128k | 30 | 35 | 56 | 34 | 60 | 0.93× |

**Qwen3-4B** (draft: l7bc, exact)

| Context | Ours before | Ours after | Ours + draft | llama.cpp | llama.cpp + its draft | Our best / their best |
|---|---|---|---|---|---|---|
| 2k | 252 | 280 | 311 | 240 | 164 | 1.30× |
| 4k | 240 | 265 | 292 | 226 | 181 | 1.29× |
| 8k | 221 | 241 | 243 | 205 | 149 | 1.19× |
| 16k | 191 | 206 | 179 | 174 | 119 | 1.19× |
| 32k | 150 | 159 | 137 | 128 | 80 | 1.24× |
| 64k | 104 | 109 | 82 | 82 | 54 | 1.33× |
| 128k | 65 | 66 | 46 | 46 | – | 1.43× |

**Inkling 2 Mini, llama.cpp Q4_K_M**

| Context | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|
| Tokens/s | 97 | 95 | 93 | 91 | 86 | 78 | 66 |
