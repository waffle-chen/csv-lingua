# One sentence, step by step

Input (105 characters):

> John: So, um, I've been thinking about the project, you know, and I believe we need to make some changes.

## 1. Plan: language, protected characters, chunks

- language rules: `default`
- protected characters: '\n' '.' '!' '?' ','
- replaced before tokenizing: {'\n': '[NEW0]'} (a newline is not a WordPiece token)
- code parts kept as they are: 0
- chunks: 1 (at most 512 tokens each)

## 2. Tokenizer

`[CLS]` and `[SEP]` are added, so this chunk has 30 tokens.

| position | token | token id |
|---|---|---|
| 0 | `[CLS]` | 101 |
| 1 | `John` | 10421 |
| 2 | `:` | 131 |
| 3 | `So` | 12882 |
| 4 | `,` | 117 |
| 5 | `um` | 10293 |
| 6 | `,` | 117 |
| 7 | `I` | 146 |
| 8 | `'` | 112 |
| 9 | `ve` | 10323 |
| 10 | `been` | 10590 |
| 11 | `thinking` | 56294 |
| … | … | … |

## 3. The CSV model

- 220 CSV files; only the 26 word-embedding rows this text needs were read
- `weights/bert.embeddings.word_embeddings.weight.part00.csv` stores row = token id, so token 10421 (`John`) is one line of that file
- its first numbers: [-0.07205, -0.04403, -0.02502, -0.01901,  0.003  ,  0.01401] … (768 in total)

## 4. BERT

- embeddings: 30 tokens × 768 numbers
- 12 layers, each: attention over 30×30 token pairs per head (12 heads of 64), then 768 → 3072 → 768
- hidden state after layer 1, first numbers: [-0.0515, -0.0112, -0.0634, -0.0544]
- hidden state after layer 12, first numbers: [ 0.1194, -1.1554,  1.6606,  0.1689]
- classifier: 768 → 2 → softmax → keep-probability per token

| token | p_keep |
|---|---|
| `[CLS]` | 0.0006 |
| `John` | 0.9545 |
| `:` | 0.4188 |
| `So` | 0.1251 |
| `,` | 0.0427 |
| `um` | 0.0586 |
| `,` | 0.0118 |
| `I` | 0.0408 |
| `'` | 0.0735 |
| `ve` | 0.0555 |
| `been` | 0.2820 |
| `thinking` | 0.9080 |
| … | … |

## 5. Tokens become words

`##` pieces join the word before them, a word's probability is the mean of its tokens, and protected punctuation gets 1.0.

| word | p_word | kept |
|---|---|---|
| `John` | 0.9545 | yes |
| `:` | 0.4188 | yes |
| `So` | 0.1251 | yes |
| `,` | 1.0000 | yes |
| `um` | 0.0586 | no |
| `,` | 0.0000 | no |
| `I` | 0.0408 | no |
| `'` | 0.0735 | no |
| `ve` | 0.0555 | no |
| `been` | 0.2820 | yes |
| `thinking` | 0.9080 | yes |
| `about` | 0.5752 | yes |
| `the` | 0.0215 | no |
| `project` | 0.9885 | yes |
| … | … | … |

## 6. Threshold

- rate 0.6 → q = int(100·(1 − 0.6) + 1) = 41
- the 41-th percentile of this chunk's word probabilities is **0.0753**
- 16 of 28 words are above it

## 7. Result

> John : So, been thinking about project, believe we need to make changes.

105 characters → 72 characters (69%).
