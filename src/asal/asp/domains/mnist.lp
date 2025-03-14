%------- %
% MNIST  %
%------- %
% Use either one of the blocks below. The second one is essentially a propositionalization approach.
% Comment-out the "equals" definition that is normally used (below in this file),
% since an equals/1 pred is provided here.

holds(equals(even,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0.
holds(equals(odd,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0.
holds(equals(gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X > 6.
holds(equals(leq_6,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 6.
holds(equals(gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X > 3.
holds(equals(leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 3.

holds(equals(gt_2,1),SeqId,T) :- seq(SeqId,d(X),T), X > 2.
holds(equals(leq_2,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 2.
holds(equals(gt_4,1),SeqId,T) :- seq(SeqId,d(X),T), X > 4.
holds(equals(leq_4,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 4.

% holds(equals(gt_5,1),SeqId,T) :- seq(SeqId,d(X),T), X > 5.
% holds(equals(leq_5,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 5.
% holds(equals(gt_7,1),SeqId,T) :- seq(SeqId,d(X),T), X > 7.
% holds(equals(leq_7,1),SeqId,T) :- seq(SeqId,d(X),T), X <= 7.

categorical(even; odd; gt_6; leq_6; gt_3; leq_3).
categorical(gt_2; leq_2; gt_4; leq_4; gt_5; leq_5; gt_7; leq_7).
value(gt_2,1). value(leq_2,1). value(gt_4,1). value(leq_4,1). value(gt_5,1).
value(leq_5,1). value(gt_7,1). value(leq_7,1).
numerical(none).
value(even,1). value(odd,1). value(gt_6,1).
value(leq_6,1). value(gt_3,1). value(leq_3,1).


%*
holds(equals(even_gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X > 6.
holds(equals(even_leq_6_gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X > 3, X <= 6.
holds(equals(even_leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 = 0, X <= 3.

holds(equals(odd_gt_6,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X > 6.
holds(equals(odd_leq_6_gt_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X > 3, X <= 6.
holds(equals(odd_leq_3,1),SeqId,T) :- seq(SeqId,d(X),T), X \ 2 != 0, X <= 3.

categorical(even_gt_6; even_leq_6_gt_3; even_leq_3; odd_gt_6; odd_leq_6_gt_3; odd_leq_3).
numerical(none).
value(even_gt_6,1). value(even_leq_6_gt_3,1). value(even_leq_3,1).
value(odd_gt_6,1). value(odd_leq_6_gt_3,1). value(odd_leq_3,1).
*%
