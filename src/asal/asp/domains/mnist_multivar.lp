% run with:
% --tclass 1 --train /home/nkatz/dev/asal/data/mnist_nesy/clingo_generated_train.lp \
% --test /home/nkatz/dev/asal/data/mnist_nesy/clingo_generated_test.lp \
%  --domain /home/nkatz/dev/asal/src/asal/asp/domains/mnist_multivar.lp \
%  --predicates equals  --tlim 60 --states 4 --batch_size 50 --mcts_iters 10 --all_opt

holds(equals(even(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X \ 2 = 0.
holds(equals(odd(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X \ 2 != 0.
holds(equals(gt_6(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X > 6.
holds(equals(leq_6(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X <= 6.
holds(equals(gt_3(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X > 3.
holds(equals(leq_3(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X <= 3.

holds(equals(gt_2(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X > 2.
holds(equals(leq_2(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X <= 2.
holds(equals(gt_4(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X > 4.
holds(equals(leq_4(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X <= 4.

holds(equals(gt_5(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X > 5.
holds(equals(leq_5(D),1),SeqId,T) :- seq(SeqId,obs(D,X),T), X <= 5.

% digit(d1; d2; d3).
digit(d1; d2).

categorical(even(X)) :- digit(X).
categorical(odd(X)) :- digit(X).
categorical(gt_6(X)) :- digit(X).
categorical(leq_6(X)) :- digit(X).
categorical(gt_3(X)) :- digit(X).
categorical(leq_3(X)) :- digit(X).
categorical(gt_2(X)) :- digit(X).
categorical(leq_2(X)) :- digit(X).
categorical(gt_4(X)) :- digit(X).
categorical(leq_4(X)) :- digit(X).
categorical(gt_5(X)) :- digit(X).
categorical(leq_5(X)) :- digit(X).

value(even(X),1) :- digit(X).
value(odd(X),1) :- digit(X).
value(gt_6(X),1) :- digit(X).
value(leq_6(X),1) :- digit(X).
value(gt_3(X),1) :- digit(X).
value(leq_3(X),1) :- digit(X).
% value(gt_2(X),1) :- digit(X).
% value(leq_2(X),1) :- digit(X).
% value(gt_4(X),1) :- digit(X).
% value(leq_4(X),1) :- digit(X).
value(gt_5(X),1) :- digit(X).
value(leq_5(X),1) :- digit(X).

numerical(none).
