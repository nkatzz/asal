
%----------------------------------
% Used for BSC ductal/lobular data.
%----------------------------------
holds(enriched(Pathway),SeqId,T) :- seq(SeqId,enriched(Pathway),T).
holds(not_enriched(Pathway),SeqId,T) :- seq(SeqId,not_enriched(Pathway),T).
pathway(P) :- seq(_, enriched(P), _).
pathway(P) :- seq(_, not_enriched(P), _).

cost(enriched(A),1) :- pathway(A).
cost(not_enriched(A),1) :- pathway(A).
