## SSCC 1st / theeluwin

Source code for paper submitted by [@theeluwin][4] at Proceedings of SSCC 1st.


Implemented algorithm that labels locale tag for each korean character using [CRF][1].

### Requirements

This source code requires [python-crfsuite][2] and [scikit-learn][3] as dependency.

```bash
$ virtualenv venv
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt
```

### Usage

With appropriate `train.txt` and `test.txt`, just run

```bash
(venv)$ python sscc.py
```

[1]: http://dl.acm.org/citation.cfm?id=655813
[2]: https://github.com/tpeng/python-crfsuite
[3]: https://github.com/scikit-learn/scikit-learn
[4]: https://twitter.com/theeluwin
