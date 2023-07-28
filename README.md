# ASAL
Answer Set Automata Learning
----------------------------

Learning symbolic automata-based complex event patterns in Answer Set Programming. See the related paper for more info:

[Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, ECAI 2023](https://cer.iit.demokritos.gr/publications/papers/2023/katz_2023.pdf)

Install Clingo by following the instructions at https://potassco.org/clingo/. The software has been tested with Python 3.10 and Clingo 5.6.2. See the requirements.txt file for additional required python libraries.

In the ```src``` folder there are two scripts from which the batch and the incremental, MCTS-based versions of ASAL can be run. The hyper-parameters/runtime arguments that control a run are explained in comments therein. The arguments can be tweaked by editing a script, since there is no CLI at this point. 

To select a particular dataset/fold to run modify the ```train_path``` and ```test_path``` variables to reflect a train and test file for a particular dataset/fold in the the ```data``` folder. After that the scripts may be run in the regular way, e.g. ```python run_mcts.py```.
<!---
To use RPNI/EDSM the LearnLib library is required: https://learnlib.de/. Follow the instructions to install the software. Then use the ```to_rpni``` method in ```src/asal/auxils.py``` to convert the input seqs to RPNI format, by providing the path to a train/test file and follow the LearnLib instructions to run the respective methods (rpni/edsm).
--->
