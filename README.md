# ASAL
Answer Set Automata Learning
----------------------------

ASAL is a framework for representing and learning symbolic automata-based complex event patterns in Answer Set Programming. See the related paper for more information on the learning task and the available data:

_Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, ECAI 2023._ ([link](https://cer.iit.demokritos.gr/publications/papers/2022/ilp-2022.pdf))

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
python run_batch.py
python run_mcts.py
```


With the two scripts above the batch and the incremental, Monte Carlo Tree Search-based versions of ASAL can be run. The hyper-parameters/runtime arguments that control a run are explained in comments in the scripts. The arguments can be tweaked by editing a script, since there is no CLI at this point. 

To select a particular dataset/fold to run modify the ```dataset```, ```fold```, ```train_path``` and ```test_path``` variables in a run script to reflect a train/test pair for a particular dataset/fold in the ```data``` folder. Again, see the paper above for info on these datasets/tasks.

<!---
To use RPNI/EDSM the LearnLib library is required: https://learnlib.de/. Follow the instructions to install the software. Then use the ```to_rpni``` method in ```src/asal/auxils.py``` to convert the input seqs to RPNI format, by providing the path to a train/test file and follow the LearnLib instructions to run the respective methods (rpni/edsm).
--->
