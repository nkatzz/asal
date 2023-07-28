# ASAL
Answer Set Automata Learning
----------------------------

ASAL is a framework for representing and learning symbolic automata-based complex event patterns in Answer Set Programming. See [the related paper](https://cer.iit.demokritos.gr/publications/papers/2022/ilp-2022.pdf) for more information on the learning task and the available data:

Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, ECAI 2023.

To use the software you will need the [Clingo](https://potassco.org/clingo) ASP solver. It is advised to check the instructions on the Clingo web page for potential changes in the first command below. 

```
conda create -n asal -c conda-forge clingo
conda activate asal
conda install pip
git clone https://github.com/nkatzz/asal.git
cd asal
pip install -r requirements.txt
```

In the ```asal/src``` folder there are two scripts from which the batch and the incremental, Monte Carlo Tree Search-based versions of ASAL can be run. The hyper-parameters/runtime arguments that control a run are explained in comments therein. The arguments can be tweaked by editing a script, since there is no CLI at this point. 

To select a particular dataset/fold to run modify the ```dataset```, ```fold```, ```train_path``` and ```test_path``` variables to reflect a train/test pair for a particular dataset/fold in the ```data``` folder. Again, see the paper above for info on these datasets/tasks. The scripts are run in the regular way, i.e. ```python run_mcts.py```.
<!---
To use RPNI/EDSM the LearnLib library is required: https://learnlib.de/. Follow the instructions to install the software. Then use the ```to_rpni``` method in ```src/asal/auxils.py``` to convert the input seqs to RPNI format, by providing the path to a train/test file and follow the LearnLib instructions to run the respective methods (rpni/edsm).
--->
