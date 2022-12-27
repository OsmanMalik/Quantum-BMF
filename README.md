# Binary matrix factorization using D-Wave's quantum annealer

This repo illustrates how to use D-Wave's quantum annealer for computing binary
matrix factorization (BMF). This is done via the optimization formulation 
proposed in the paper
> O. A. Malik, H. Ushijima-Mwesigwa, A. Roy, A. Mandal, I. Ghosh. 
*Binary matrix factorization on special purpose hardware*. 
PLOS ONE 16(12): e0261250, 2021. DOI: 
[10.1371/journal.pone.0261250](https://doi.org/10.1371/journal.pone.0261250)

Please be aware that the code in this repo is not indended to recreate the 
results in the paper above. The paper includes a more extensive set of 
experiments which were run on Fujitu's Digital Annealer.

## Referencing this code

Please cite our paper when appropriate:

```
@article{10.1371/journal.pone.0261250,
    doi = {10.1371/journal.pone.0261250},
    author = {Malik, Osman Asif AND Ushijima-Mwesigwa, Hayato AND Roy, Arnab AND Mandal, Avradip AND Ghosh, Indradeep},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Binary matrix factorization on special purpose hardware},
    year = {2021},
    month = {12},
    volume = {16},
    url = {https://doi.org/10.1371/journal.pone.0261250},
    pages = {1-18},
    number = {12},
}
```

## Some further details on the code

The notebook `demo_notebook.ipynb` provides a brief introduction to BMF and 
shows how to run BMF using different D-Wave samplers. In order to use the 
hybrid and quantum samplers, you will need a D-Wave Leap account. It is 
straightforward to set up a free trial account at https://cloud.dwavesys.com.

## Author contact information

Please feel free to contact me at any time if you have any questions or would 
like to provide feedback on this code or on the paper. I can be reached at 
`oamalik (at) lbl (dot) gov`.
