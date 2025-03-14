# ASAL
Answer Set Automata Learning
----------------------------

ASAL is a framework for representing and learning symbolic automata-based complex event patterns in Answer Set Programming. 
Symbolic automata are an extention of classical automata, where the transition-enabling conditions are logical predicates 
than need to be evaluated against the input, rather than mere symbols from a finite alphabet. See the following related paper for more information:

_Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, ECAI 2023._ ([link](https://cer.iit.demokritos.gr/publications/papers/2023/ecai2023.pdf))

To use the software you will need the [Clingo ASP solver](https://potassco.org/clingo). The following set of commands create a fresh conda environment, called ```asal``` and install Clingo there, along with some additional requirements. 

```
conda create -n asal -c conda-forge clingo
conda activate asal
conda install pip
git clone https://github.com/nkatzz/asal.git
cd asal
pip install -r requirements.txt
```

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
