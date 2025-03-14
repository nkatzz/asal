inState(SeqId,1,T) :- sequence(SeqId), seqStart(SeqId,T).
inState(SeqId,S2,T+1) :- inState(SeqId,S1,T), transition(S1,F,S2), holds(F,SeqId,T).
accepted(SeqId) :- inState(SeqId,S,T), accepting(S), seqEnd(SeqId,T).
% accepted(SeqId) :- inState(SeqId,S,T), accepting(S).
reach_accept_at(SeqId,T) :- inState(SeqId,S,T), accepting(S); #false: inState(SeqId,S,T1), T1 < T.

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T+1) :- seq(SeqId,T), not seq(SeqId,T+1).
seqStart(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T-1).