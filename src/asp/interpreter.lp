inState(SeqId,1,0) :- sequence(SeqId).
inState(SeqId,S2,T+1) :- inState(SeqId,S1,T), transition(S1,F,S2), holds(F,SeqId,T).
accepted(SeqId) :- inState(SeqId,S,T), accepting(S), seqEnd(SeqId,T).
reach_accept_at(SeqId,T) :- inState(SeqId,S,T), accepting(S); #false: inState(SeqId,S,T1), T1 < T.

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T+1) :- seq(SeqId,T), not seq(SeqId,T+1).