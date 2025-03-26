# ASAL
Answer Set Automata Learning
----------------------------

ASAL is a framework for representing and learning symbolic automata-based complex event patterns in Answer Set Programming. 
Symbolic automata are an extention of classical automata, where the transition-enabling conditions are logical predicates 
than need to be evaluated against the input, rather than mere symbols from a finite alphabet. See the following related paper for more information:

_Katzouris N. & Paliouras G., Answer Set Automata: A Learnable Pattern Specification Framework for Complex Event Recognition, 
ECAI 2023._ ([link](https://cer.iit.demokritos.gr/publications/papers/2023/ecai2023.pdf))

## Installation
You will need Python 3.11 and the [Clingo ASP solver](https://potassco.org/clingo). You get Python-enabled Clingo by installing the requirements file.
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

For the neuro-symbolic version of ASAL (see below) you will additionally need the [dsharp](https://github.com/QuMuLab/dsharp) tool for knowledge 
compilation. You need to make it and adding the ```dhsharp``` folder to your ```PATH``` variable.
## Usage

```
python asal.py --help

usage: args_parser.py [-h] [--train <path>] --domain <path> [--test <path>]
                      [--tlim <n>] [--states <n>] [--tclass <n>]
                      [--unsat_weight <n>] [--incremental] [--batch_size <n>]
                      [--mcts_iters <n>] [--exp_rate <float>]
                      [--mcts_children <n>] [--max_alts <n>]
                      [--coverage_first] [--min_attrs] [--all_opt] --eval
                      <path> [--show <s|r>] [--warns_off]
                      [--predicates {equals,neg,lt,at_most,at_least,increase,decrease} [{equals,neg,lt,at_most,at_least,increase,decrease} ...]]

options:
  -h, --help            show this help message and exit
  --train <path>        path to the training data.
  --domain <path>       path to the domain specification file.
  --test <path>         path to the testing data.
  --tlim <n>            time limit for Clingo in secs [default: 0, no limit].
  --states <n>          max number of states in a learnt automaton [default: 3].
  --tclass <n>          target class to predict (one-vs-rest) [default: 1].
  --unsat_weight <n>    penalty for not accepting a positive sequence, or rejecting 
                        a negative one. The default weight is 1 and is applied 
                        uniformly to  all training sequences. Individual weights 
                        per example can be set via --unsat_weight 0, in which case the 
                        weights need to be provided in the training data file as weight(S,W)
                        where S is the sequence id and W is an integer.
  --incremental         learn incrementally with MCTS.
  --batch_size <n>      mini batch size for incremental learning.
  --mcts_iters <n>      number of MCTS iterations for incremental learning.
  --exp_rate <float>    exploration rate for MCTS in incremental learning.
  --mcts_children <n>   number of children nodes to consider in MCTS.
  --max_alts <n>        max number of disjunctive alternatives per transition guard.
  --coverage_first      set a higher priority to constraints that minimize FPs & FNs 
                        over constraints that minimize model size.
  --min_attrs           minimize the number of different attributes/predicates that appear in a model.
  --all_opt             find all optimal models during Clingo search.
  --eval <path>         path to a file that contains an SFA specification (learnt/hand-crafted).
                        to evaluate on test data (passed via the --test option). The automaton needs to be
                        in reasoning-based format (see option --show)
  --show <s|r>          show learnt SFAs in simpler (s), easier to inspect format, 
                        or in a format that can be used for reasoning (r) with Clingo.
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

### Example
The following learns a symbolic automaton, represented as an ASP program, from symbolic sequences of overtake incidents
in the [ROAD-R](https://sites.google.com/view/road-r/) dataset. The task is described in the following MSc thesis:

_Tatiana Boura, Neuro-symbolic Complex Event Recognition in Autonomous Driving, University of Piraeus, 2024._ ([link](https://msc-ai.iit.demokritos.gr/en/thesis/neuro-symbolic-complex-event-recognition-autonomous-driving-neyro-symvoliki-anagnorisi?page=2))

```
cd asal/src
python asal.py --tlim 60 --states 4 --tclass 2 --train ../data/ROAD-R/folds/split_9/agent_train.txt \
--test ../data/ROAD-R/folds/split_9/agent_test.txt --domain asal/asp/domains/domain.lp --batch_size 200 --predicates equals
```
The induced SFA looks like this:

```
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(1,f(1,4),4). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(2,f(2,4),4). transition(3,f(3,2),2). transition(3,f(3,3),3). transition(4,f(4,4),4).
f(1,4) :- equals(same_lane,true), equals(action_2,movaway).
f(1,3) :- equals(same_lane,false), not f(1,4).
f(1,2) :- equals(action_1,stop), not f(1,3), not f(1,4).
f(2,3) :- equals(action_1,stop), not f(2,4).
f(2,4) :- equals(action_2,movtow), equals(location_2,incomlane).
f(3,2) :- equals(action_2,movaway).
f(4,4) :- #true.
f(2,2) :- not f(2,3), not f(2,4).
f(1,1) :- not f(1,2), not f(1,3), not f(1,4).
f(3,3) :- not f(3,2).
```
To use the SFA for temporal reasoning (i.e. processing sequences), it is necessary to have it in "reasoning" format. To
do so, run ASAL with option ```--show r```. This time the result looks something like this:

```
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(1,f(1,3),3). transition(2,f(2,2),2). transition(2,f(2,3),3). 
transition(2,f(2,4),4). transition(3,f(3,3),3). transition(3,f(3,4),4). transition(4,f(4,4),4).
holds(f(1,2),S,T) :- holds(equals(action_1,stop),S,T), holds(equals(action_2,movtow),S,T), not holds(f(1,3),S,T).
holds(f(2,3),S,T) :- holds(equals(action_1,stop),S,T), not holds(f(2,4),S,T).
holds(f(3,4),S,T) :- holds(equals(action_1,movaway),S,T), holds(equals(location_2,vehlane),S,T).
holds(f(1,3),S,T) :- holds(equals(location_1,vehlane),S,T).
holds(f(2,4),S,T) :- holds(equals(location_2,incomlane),S,T).
holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,4),S,T).
holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,2),S,T), not holds(f(1,3),S,T).
holds(f(4,4),S,T) :- sequence(S), time(T).
holds(f(2,2),S,T) :- sequence(S), time(T), not holds(f(2,3),S,T), not holds(f(2,4),S,T).
```
This can be used to classify sequences. For instance, by writing the automaton above to a file 
(```path/to/sfa``` in the command below) and  running (note the ```--eval``` option):

```
cd asal/src
python asal.py --tclass 2 --test ../data/ROAD-R/folds/split_9/agent_test.txt --domain asal/asp/domains/roadr.lp \
 --eval path/to/sfa --predicates equals 
```
the SFA is evaluated on the test data.
<!---
There are several other datasets in the ```data``` folder and domain specifications for each in ```src/asal/asp/domains```.
-->

### Input
We next proceed to the description of the ASAL input data, i.e. the way that the content of the files passed via the
 ```--train``` and ```--test``` options should be formatted.

ASAL learns from (possibly multivariate) symbolic sequences, represented as a set of ASP facts. These facts are of the form
```seq(SeqId,Pred,Time)```, where ```SeqId``` is unique sequence identifier, ```T``` is a time point (integer) and
```Pred``` is a ground predicate, or attribute/value pair that holds (is true) at time ```T``` in sequence ```SeqId```. 
For instance, the atom 
```seq(1,a1(action(movtow)),3).``` dictates that a1 (a detected vehicle in ROAD-R) is performing action ```movtow```
(moving towards the main vehicle's camera) at time point 3 in sequence 1. Multivariate sequences are represented in the 
same fashion. For instance, ```seq(1, a2(action(movaway)), 5).``` dictates that in the same sequence (with id 1) vehicle 
a2 is moving away from the main vehicle at time 5. The symbolic signals (i.e. sequences of ```seq```) for a1 and a2's actions, locations
and bounding box coordinates over time, form the symbolic representation of the multivariate sequence with id 1. 
For instance, the length-three prefix of an actual ```ROAD-R``` sequence is show below:

```
agent_type(685, agent1(larveh)). agent_type(685, agent2(car)). class(685, 2).
seq(685, a1(action(stop)), 1). seq(685, a1(action(stop)), 2). seq(685, a1(action(stop)), 3). class(685, 2).
seq(685, a2(action(movtow)), 1). seq(685, a2(action(movtow)), 2). seq(685, a2(action(movtow)), 3). class(685, 2).
seq(685, a1(location(incomlane)), 1). seq(685, a1(location(incomlane)), 2). seq(685, a1(location(incomlane)), 3). class(685, 2).
seq(685, a2(location(incomlane)), 1). seq(685, a2(location(incomlane)), 2). seq(685, a2(location(incomlane)), 3). class(685, 2).
seq(685, a1(xcoord(x1, "0.8390187771520882")), 1). seq(685, a1(xcoord(x1, "0.8312306151583189")), 2). seq(685, a1(xcoord(x1, "0.8234424531645492")), 3). class(685, 2).
seq(685, a1(xcoord(x2, "0.8978971677405811")), 1). seq(685, a1(xcoord(x2, "0.8951713110427617")), 2). seq(685, a1(xcoord(x2, "0.8924454543449425")), 3). class(685, 2). 
seq(685, a1(ycoord(y1, "0.37258568879599885")), 1). seq(685, a1(ycoord(y1, "0.37154726719682957")), 2). seq(685, a1(ycoord(y1, "0.37050884559766034")), 3). class(685, 2).
seq(685, a1(ycoord(y2, "0.532710280373832")), 1). seq(685, a1(ycoord(y2, "0.5316718587746627")), 2). seq(685, a1(ycoord(y2, "0.5306334371754935")), 3). class(685, 2).
seq(685, a2(xcoord(x1, "0.7387850752500728")), 1). seq(685, a2(xcoord(x1, "0.7309969132563033")), 2). seq(685, a2(xcoord(x1, "0.7232087512625338")), 3). class(685, 2).
seq(685, a2(xcoord(x2, "0.8156542911707797")), 1). seq(685, a2(xcoord(x2, "0.8090343534760758")), 2). seq(685, a2(xcoord(x2, "0.8024144157813715")), 3). class(685, 2).
seq(685, a2(ycoord(y1, "0.44828663288247184")), 1). seq(685, a2(ycoord(y1, "0.4477674220828872")), 2). seq(685, a2(ycoord(y1, "0.44724821128330255")), 3). class(685, 2).
seq(685, a2(ycoord(y2, "0.5345794582664039")), 1). seq(685, a2(ycoord(y2, "0.5366563014647424")), 2). seq(685, a2(ycoord(y2, "0.5387331446630809")), 3). class(685, 2).
```

Note that the ```seqId``` in each line is the same, indicating that these are sub-sequences of the same multivariate
sequence. Note also that each line ends with a ```class(SeqId,ClassLabel)``` atom, that provides a label for the entire
sequence. It is currently mandatory to have each line in the symbolic representation of a multivariate
sequence end with such an atom.

### Domain Specification
The domain file provides background knowledge and a language bias for learning. We use the ```ROAD-R``` 
domain specification to explain the main structure and language bias definition. 
The presentation refers to and follows the  data format in the ```ROAD-R``` data (see the data folder).

We use the ```attribute/1``` predicate to specify symbols that can be used to 
synthesize transition guard rules in an SFA. Any symbol wrapped inside this predicate is added to the 
language bias and can be used in the bodies of transition guards. These symbols
either refer directly to data attributes, or are defined as background knowledge predicates 
(see the section on custom Predicates below for the latter).

For instance, the attributes below are meant to refer to the actions and locations of the two 
vehicles that are tracked in ```ROAD-R``` data sequences. 

```attribute(action_1 ; action_2 ; location_1 ; location_2). ```

Attributes extracted directly from the data need to be wrapped in an ```obs/2``` predicate
(which stands for "observation") as in: 

```seq(seq_id,obs(action_1,movaway),15).```

If the data are not in such format, we can either convert them into it, or use rules such as the following ones to 
transform the data on the fly (note that the bodies of these rules are the ```seq/3``` signatures, as
they appear in the data, e.g. ```seq(1,a2(action(movaway)),5).```):
```
seq(SeqId,obs(action_1,X),T) :- seq(SeqId,a1(action(X)),T), allowed_actions(X).
seq(SeqId,obs(action_2,X),T) :- seq(SeqId,a2(action(X)),T), allowed_actions(X).
seq(SeqId,obs(location_1,X),T) :- seq(SeqId,a1(location(X)),T), allowed_locations(X).
seq(SeqId,obs(location_2,X),T) :- seq(SeqId,a2(location(X)),T), allowed_locations(X).
allowed_actions(stop ; movtow ; movaway).
allowed_locations(incomlane ; jun ; vehlane).
```
Here ```allowed_actions/1``` and ```allowed_locations/1``` are used to restrict the range of 
values that the target attributes can take, in order to make learning more efficient.
If these are omitted in the rules above, all action/value pairs that appear in ```seq/3``` atoms in the data
will be considered during learning, and many of these values may be irrelevant.

We use the ```value/1``` predicate to associate target attributes with allowed values:
```
value(action_1,V) :- seq(_,a1(action(V)),_), allowed_actions(V).
value(action_2,V) :- seq(_,a2(action(V)),_), allowed_actions(V).
value(location_1,V) :- seq(_,a1(location(V)),_), allowed_locations(V).
value(location_2,V) :- seq(_,a2(location(V)),_), allowed_locations(V).  
```

Again, the bodies of these rules follow the ```seq/3``` signature, as it appears in the data file. 
Of course, the attribute/value associations can also be specified explicitly, as in 

```value(action_1,moveaway). value(action_2,movtow). value(location_1,vehlane).``` 

and so on.

Target attributes needs to be declared as either categorical, or numerical, e.g.:

```
categorical(action_1 ; action_2 ; location_1 ; location_2).
numerical(xcoord_1_2; xcoord_2_1).
```
Here ```xcoord_1_2, xcoord_2_1``` refer to the ```x1``` coordinates of the two vehicles' bounding boxes, 
which could also be included in the language bias as potentially informative attributes.

Categorical attributes are input to the ```equals``` predicate, allowing to learn transition guard rules such as

```f(1,2) :- equals(action_2,movtow), equals(location_2,incomlane).```

Numerical attributes are input to comparison predicates, such as ```at_least(Attribute, Threshold)```, 
```lt(Attribute_1, Attribute_2)``` and so on. For instance, by declaring additionally:

```
attribute(xcoord_1_1). attribute(xcoord_2_1).
numerical(xcoord_1_2; xcoord_2_1).
seq(SeqId,obs(xcoord_1_2,X),T) :- seq(SeqId,a1(xcoord(x1,X)),T).
seq(SeqId,obs(xcoord_2_1,X),T) :- seq(SeqId,a2(xcoord(x1,X)),T).
``` 
and running ASAL with the option ```--predicates equals lt```, it is possible to learn transition rules such as  

```f(1,2) :- equals(action_2,movtow), lt(xcoord_1_1,xcoord_2_1).```

Numerical attributes can also be symbols, which are compared by their lexicographical order. This is useful when such
attributes represent e.g. bins of discretized numerical values. 

To reason with attribute/value pair predicates over time, ASAL uses the ```holds/3``` predicate, with signature
```holds(Predicate, SeqId, Time)```. This predicate is used to define when a fluent (temporal property) holds 
at particular points over time in a sequence.
```holds/3``` definitions for the basic predicates (```equals, lt, at_least, at_most, neg``` etc) are generated at
runtime. For instance, the following rules are internally added to the domain, if ASAL is run with e.g ```--predicates equals lt```:

```
holds(equals(A,X),SeqId,T) :- seq(SeqId,obs(A,X),T), categorical(A).
holds(lt(A1,A2),SeqId,T) :- seq(SeqId,obs(A1,X),T), seq(SeqId,obs(A2,Y),T), X < Y, numerical(A1), numerical(A2), A1 != A2.
```
In the bodies of these rules, note the ```obs/2``` predicate discussed earlier, for capturing attribute/value pairs.
Because target predicate (e.g ```equals```) fluents are defined via ```obs/2```, this is why the data should adhere to
this representation, as explained above.

#### Adding Custom Predicates to the Language Bias

In addition to the basic predicates and the data-extracted feature/value pairs, arbitrary predicates can be
defined via ```holds/3```, directly inside the domain file. This can be achieved by viewing such predicates as boolean-valued
domain features. For instance, the following rules define a new predicate, which is true when the two vehicles are on
the same lane:

```
holds(equals(same_lane,true),SeqId,T) :- seq(SeqId,obs(location_1,X),T), seq(SeqId,obs(location_2,X),T).
holds(equals(same_lane,false),SeqId,T) :- not holds(equals(same_lane,true),SeqId,T).
``` 

We can include this predicate in the language bias and allow to learn rules such as: ```f(1,2) :- equals(same_lane,true).```
by adding to the domain file: 

```attribute(same_lane). categorical(same_lane). value(same_lane,true). value(same_lane,false).```

### Domain Specification summary:

1. Define the symbols that you want to appear in transition guard rules via the ```attribute/1 predicate```. 
2. Define the values associated with each attribute via the ```value/2``` predicate.
3. Define custom predicates via ```holds/3```, by treating such predicates as boolean-valued domain features. Include such
predicates to the language bias by declaring them as target symbols, as in 1, 2.
4. Ensure that the data adhere to the ```obs/2``` attribute/value representation and your custom predicate definitions
use ```obs/2``` in the definitions of fluents.
5. Allow for the basic predicates via the ```--predicates``` option (see above). Allow for custom predicates by simply
adding their ```equals```-based definitions into the domain file. Note that ```[equals]``` is the default value for
the ```--predicates``` option.

## Neuro-symbolic ASAL
Description Coming soon. See the ```arc/neurasal.py``` script. In addition to the libs in requirements.txt, 
you will need the [dsharp](https://github.com/QuMuLab/dsharp)
 tool for knowledge compilation, it involves a simple ```make``` and adding the ```dhsharp``` folder to your path. 
Neuro-symbolic training is based on the following paper:

_Nikolaos Manginas, George Paliouras, Luc De Raedt., NeSyA: Neurosymbolic Automata, arxiv 2024._ ([link](https://arxiv.org/abs/2412.07331))

and the implementation from that paper of probabilistic reasoning with symbolic automata via compilation to arithmetic circuits. 

## Example
The following performs neuro-symbolic training with a toy SFA on a temporal MNIST arithmetic task (the input is sequences of MNIST images which do, or do not satisfy
a temporal pattern expressed as an SFA):

```
cd asal/src
python neurasal.py --tlim 60 --states 4 --tclass 1 --train ../data/mnist_nesy/train.csv --test ../data/mnist_nesy/test.csv --domain asal/asp/domains/mnist.lp --batch_size 200 --coverage_first
```

<!---
To use RPNI/EDSM the LearnLib library is required: https://learnlib.de/. Follow the instructions to install the software. Then use the ```to_rpni``` method in ```src/asal/auxils.py``` to convert the input seqs to RPNI format, by providing the path to a train/test file and follow the LearnLib instructions to run the respective methods (rpni/edsm).
--->
