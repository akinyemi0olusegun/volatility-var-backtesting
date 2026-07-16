| asset   | model                 | confidence   |    N |   exc | breach%   | exp%   |   Kupiec LR |   Kupiec p | Kupiec   |   CC p | Basel   |
|:--------|:----------------------|:-------------|-----:|------:|:----------|:-------|------------:|-----------:|:---------|-------:|:--------|
| EW      | EWMA (lambda=0.94)    | 95%          | 2264 |   108 | 4.77%     | 5%     |        0.26 |      0.613 | PASS     |  0.755 | n/a     |
| EW      | GARCH(1,1)-Normal     | 95%          | 2264 |   108 | 4.77%     | 5%     |        0.26 |      0.613 | PASS     |  0.878 | n/a     |
| EW      | GARCH(1,1)-t          | 95%          | 2264 |   112 | 4.95%     | 5%     |        0.01 |      0.908 | PASS     |  0.816 | n/a     |
| EW      | Historical sim (250d) | 95%          | 2264 |   119 | 5.26%     | 5%     |        0.31 |      0.579 | PASS     |  0.293 | n/a     |
| EW      | Normal (rolling 250d) | 95%          | 2264 |   110 | 4.86%     | 5%     |        0.1  |      0.757 | PASS     |  0.011 | n/a     |
| EW      | EWMA (lambda=0.94)    | 99%          | 2264 |    46 | 2.03%     | 1%     |       18.75 |      0     | FAIL     |  0     | green   |
| EW      | GARCH(1,1)-Normal     | 99%          | 2264 |    48 | 2.12%     | 1%     |       21.71 |      0     | FAIL     |  0     | green   |
| EW      | GARCH(1,1)-t          | 99%          | 2264 |    34 | 1.5%      | 1%     |        4.99 |      0.026 | FAIL     |  0.004 | green   |
| EW      | Historical sim (250d) | 99%          | 2264 |    30 | 1.33%     | 1%     |        2.19 |      0.139 | PASS     |  0.001 | green   |
| EW      | Normal (rolling 250d) | 99%          | 2264 |    50 | 2.21%     | 1%     |       24.85 |      0     | FAIL     |  0     | yellow  |
| GLD     | EWMA (lambda=0.94)    | 95%          | 2264 |   115 | 5.08%     | 5%     |        0.03 |      0.863 | PASS     |  0.126 | n/a     |
| GLD     | GARCH(1,1)-Normal     | 95%          | 2264 |   114 | 5.04%     | 5%     |        0.01 |      0.939 | PASS     |  0.864 | n/a     |
| GLD     | GARCH(1,1)-t          | 95%          | 2264 |   120 | 5.3%      | 5%     |        0.42 |      0.516 | PASS     |  0.651 | n/a     |
| GLD     | Historical sim (250d) | 95%          | 2264 |   122 | 5.39%     | 5%     |        0.7  |      0.402 | PASS     |  0.693 | n/a     |
| GLD     | Normal (rolling 250d) | 95%          | 2264 |   119 | 5.26%     | 5%     |        0.31 |      0.579 | PASS     |  0.667 | n/a     |
| GLD     | EWMA (lambda=0.94)    | 99%          | 2264 |    42 | 1.86%     | 1%     |       13.36 |      0     | FAIL     |  0.001 | green   |
| GLD     | GARCH(1,1)-Normal     | 99%          | 2264 |    39 | 1.72%     | 1%     |        9.82 |      0.002 | FAIL     |  0.004 | green   |
| GLD     | GARCH(1,1)-t          | 99%          | 2264 |    26 | 1.15%     | 1%     |        0.48 |      0.488 | PASS     |  0.581 | green   |
| GLD     | Historical sim (250d) | 99%          | 2264 |    33 | 1.46%     | 1%     |        4.2  |      0.041 | FAIL     |  0.005 | green   |
| GLD     | Normal (rolling 250d) | 99%          | 2264 |    41 | 1.81%     | 1%     |       12.13 |      0     | FAIL     |  0.001 | yellow  |
| QQQ     | EWMA (lambda=0.94)    | 95%          | 2264 |   136 | 6.01%     | 5%     |        4.55 |      0.033 | FAIL     |  0.098 | n/a     |
| QQQ     | GARCH(1,1)-Normal     | 95%          | 2264 |   138 | 6.1%      | 5%     |        5.36 |      0.021 | FAIL     |  0.067 | n/a     |
| QQQ     | GARCH(1,1)-t          | 95%          | 2264 |   142 | 6.27%     | 5%     |        7.16 |      0.007 | FAIL     |  0.026 | n/a     |
| QQQ     | Historical sim (250d) | 95%          | 2264 |   125 | 5.52%     | 5%     |        1.25 |      0.263 | PASS     |  0.165 | n/a     |
| QQQ     | Normal (rolling 250d) | 95%          | 2264 |   134 | 5.92%     | 5%     |        3.81 |      0.051 | PASS     |  0.004 | n/a     |
| QQQ     | EWMA (lambda=0.94)    | 99%          | 2264 |    52 | 2.3%      | 1%     |       28.14 |      0     | FAIL     |  0     | yellow  |
| QQQ     | GARCH(1,1)-Normal     | 99%          | 2264 |    51 | 2.25%     | 1%     |       26.48 |      0     | FAIL     |  0     | yellow  |
| QQQ     | GARCH(1,1)-t          | 99%          | 2264 |    36 | 1.59%     | 1%     |        6.75 |      0.009 | FAIL     |  0.03  | yellow  |
| QQQ     | Historical sim (250d) | 99%          | 2264 |    36 | 1.59%     | 1%     |        6.75 |      0.009 | FAIL     |  0.011 | green   |
| QQQ     | Normal (rolling 250d) | 99%          | 2264 |    67 | 2.96%     | 1%     |       57.55 |      0     | FAIL     |  0     | yellow  |
| SPY     | EWMA (lambda=0.94)    | 95%          | 2264 |   125 | 5.52%     | 5%     |        1.25 |      0.263 | PASS     |  0.26  | n/a     |
| SPY     | GARCH(1,1)-Normal     | 95%          | 2264 |   125 | 5.52%     | 5%     |        1.25 |      0.263 | PASS     |  0.091 | n/a     |
| SPY     | GARCH(1,1)-t          | 95%          | 2264 |   140 | 6.18%     | 5%     |        6.23 |      0.013 | FAIL     |  0.008 | n/a     |
| SPY     | Historical sim (250d) | 95%          | 2264 |   121 | 5.34%     | 5%     |        0.55 |      0.457 | PASS     |  0     | n/a     |
| SPY     | Normal (rolling 250d) | 95%          | 2264 |   124 | 5.48%     | 5%     |        1.05 |      0.305 | PASS     |  0.001 | n/a     |
| SPY     | EWMA (lambda=0.94)    | 99%          | 2264 |    56 | 2.47%     | 1%     |       35.21 |      0     | FAIL     |  0     | yellow  |
| SPY     | GARCH(1,1)-Normal     | 99%          | 2264 |    50 | 2.21%     | 1%     |       24.85 |      0     | FAIL     |  0     | yellow  |
| SPY     | GARCH(1,1)-t          | 99%          | 2264 |    35 | 1.55%     | 1%     |        5.84 |      0.016 | FAIL     |  0.016 | green   |
| SPY     | Historical sim (250d) | 99%          | 2264 |    40 | 1.77%     | 1%     |       10.95 |      0.001 | FAIL     |  0     | green   |
| SPY     | Normal (rolling 250d) | 99%          | 2264 |    66 | 2.92%     | 1%     |       55.36 |      0     | FAIL     |  0     | yellow  |
| TLT     | EWMA (lambda=0.94)    | 95%          | 2264 |   125 | 5.52%     | 5%     |        1.25 |      0.263 | PASS     |  0.385 | n/a     |
| TLT     | GARCH(1,1)-Normal     | 95%          | 2264 |   119 | 5.26%     | 5%     |        0.31 |      0.579 | PASS     |  0.472 | n/a     |
| TLT     | GARCH(1,1)-t          | 95%          | 2264 |   124 | 5.48%     | 5%     |        1.05 |      0.305 | PASS     |  0.409 | n/a     |
| TLT     | Historical sim (250d) | 95%          | 2264 |   122 | 5.39%     | 5%     |        0.7  |      0.402 | PASS     |  0.294 | n/a     |
| TLT     | Normal (rolling 250d) | 95%          | 2264 |   105 | 4.64%     | 5%     |        0.64 |      0.424 | PASS     |  0.151 | n/a     |
| TLT     | EWMA (lambda=0.94)    | 99%          | 2264 |    32 | 1.41%     | 1%     |        3.46 |      0.063 | PASS     |  0.112 | green   |
| TLT     | GARCH(1,1)-Normal     | 99%          | 2264 |    27 | 1.19%     | 1%     |        0.8  |      0.371 | PASS     |  0.084 | green   |
| TLT     | GARCH(1,1)-t          | 99%          | 2264 |    22 | 0.97%     | 1%     |        0.02 |      0.892 | PASS     |  0.058 | green   |
| TLT     | Historical sim (250d) | 99%          | 2264 |    30 | 1.33%     | 1%     |        2.19 |      0.139 | PASS     |  0.008 | green   |
| TLT     | Normal (rolling 250d) | 99%          | 2264 |    28 | 1.24%     | 1%     |        1.19 |      0.275 | PASS     |  0.009 | green   |