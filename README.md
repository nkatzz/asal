# ASAL
Answer Set Automata Learning
----------------------------

ASAL is a framework for representing and learning symbolic automata-based complex event patterns in Answer Set Programming. 
Symbolic automata are an extention of classical automata, where the transition-enabling conditions are logical predicates 
than need to be evaluated against the input, rather than mere symbols from a finite alphabet. See the following related paper for more information:

_Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, 
ECAI 2023._ ([link](https://cer.iit.demokritos.gr/publications/papers/2023/ecai2023.pdf))

## Installation
You will need Python 3.11 and [Clingo ASP solver](https://potassco.org/clingo). You get Python-enabled Clingo by installing the requirements file.
The latter contains additional libraries (e.g. pytorch), mostly for neuro-symbolic training. The following will generate
a new environment named ```asal``` with everything:

```
conda create -n asal python=3.11 -c conda-forge clingo
conda activate asal
conda install pip
git clone https://github.com/nkatzz/asal.git
cd asal
pip install -r requirements.txt
```

## Usage

```
python asal.py --help

usage: asal.py [-h] [--tlim <n>] [--states <n>] [--tclass <n>] --train <path> [--test <path>] --domain <path> [--incremental] [--batch_size <n>] [--max_alts <n>] [--mcts_iters <n>] [--exp_rate <float>]
               [--mcts_children <n>] [--coverage_first] [--min_attrs] [--warns_off] --predicates {equals,neg,lt,at_most,at_least,increase,decrease} [{equals,neg,lt,at_most,at_least,increase,decrease} ...]

options:
  -h, --help            show this help message and exit
  --tlim <n>            time limit for Clingo in secs (default: 0, no limit).
  --states <n>          max number of states in a learnt automaton.
  --tclass <n>          target class to predict (one-vs-rest).
  --train <path>        path to the training data.
  --test <path>         path to the testing data.
  --domain <path>       path to the domain specification file.
  --incremental         learn incrementally with MCTS.
  --batch_size <n>      mini batch size for incremental learning.
  --max_alts <n>        max number of disjunctive alternatives per transition guard.
  --mcts_iters <n>      number of MCTS iterations for incremental learning.
  --exp_rate <float>    exploration rate for MCTS in incremental learning.
  --mcts_children <n>   number of children nodes to consider in MCTS.
  --coverage_first      higher priority to predictive performance optimization constraints over model size ones.
  --min_attrs           minimize the number of attributes that appear in a model.
  --warns_off           suppress warnings from Clingo.
  --predicates {equals,neg,lt,at_most,at_least,increase,decrease} [{equals,neg,lt,at_most,at_least,increase,decrease} ...]
                        List of predicates to use for synthesizing transition guards. 
                        These are necessary for adding proper generate and test 
                        statements to the ASP induction program. 
                                
                        Example usage: --predicates equals neg at_most        
                        Options: 
                            - equals:   allows atoms of the form equals(A,V) in the transition guards, meaning that attribute A has value V. 
                                        The available attribute/value pairs can be extracted directly from the data, or defined via rules in the 
                                        background knowledge provided with the domain specification file. See available examples in the README.
                                        attribute A must be declared as 'categorical' in the domain specification.   
                            - neg:      allows atoms of the form neg(A,V) in the transition guards, meaning that attribute A does not have value V.
                            - lt:       allows atoms of the form lt(A1,A2) in the transition guards, meaning that the value of attribute A1 is 
                                        smaller than the value of A2. 
                                        Attributes A1, A2 must be declared as 'numerical' in the domain specification.
                            - at_most:  allows atoms of the form at_most(A,V) in the transition guards, meaning that the value of attribute A is 
                                        smaller than V. Attribute A must be declared as 'numerical' in the domain specification.
                            - at_least: allows atoms of the form at_least(A,V) in the transition guards, meaning that the value of attribute A is 
                                        larger than V. Attribute A must be declared as 'numerical' in the domain specification.
                            - increase: allows atoms of the form increase(A) in the transition guards, meaning that the value of attribute A has
                                        increased since the last time step. Attribute A must be declared as 'numerical' in the domain specification.
                            - decrease: allows atoms of the form decrease(A) in the transition guards, meaning that the value of attribute A has
                                        decreased since the last time step. Attribute A must be declared as 'numerical' in the domain specification.
```

## Neuro-symbolic ASAL
Description Coming soon. See the ```arc/neurasal.py``` script. In addition to the libs in requirements.txt, 
you will need the [dsharp](https://github.com/QuMuLab/dsharp)
 tool for knowledge compilation, it involves a simple ```make``` and adding the ```dhsharp``` folder to your path. 
Neuro-symbolic training is based on the following paper:

_Nikolaos Manginas, George Paliouras, Luc De Raedt., NeSyA: Neurosymbolic Automata, arxiv 2024._ ([link](https://arxiv.org/abs/2412.07331))

and the implementation from that paper of probabilistic reasoning with symbolic automata via compilation to arithmetic circuits. 







Running the software:
```
conda activate asal
cd asal/src
python asal_batch.py
python asal_mcts.py
```


With the two scripts above (```asal_batch.py```, ```asal_mcts.py```) the batch and the incremental, Monte Carlo Tree Search-based versions of ASAL can be run. The hyper-parameters/runtime arguments that control a run are explained in comments in the scripts. The arguments can be tweaked by editing a script, since there is no CLI at this point. 

A batch run can be simulated by the ```asal_mcts.py``` script, by setting ```mini_batch_size = n```, where ```n``` is a number larger than the number of sequences in the training set and ```mcts_iterations = 1```. Therefore, the ```asal_batch.py``` script is somewhat redundant. Its main purpose, however, is to showcase the barebones process by which a model can be learned and evaluated. 

To select a particular dataset/fold to run modify, the ```dataset``` and ```fold``` variables in the run scripts. See the paper above for info on (some of) the datasets in the ```data``` folder.

<!---
To use RPNI/EDSM the LearnLib library is required: https://learnlib.de/. Follow the instructions to install the software. Then use the ```to_rpni``` method in ```src/asal/auxils.py``` to convert the input seqs to RPNI format, by providing the path to a train/test file and follow the LearnLib instructions to run the respective methods (rpni/edsm).
--->
