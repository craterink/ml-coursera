function cost = logCost(y, hyp)

cost = -(y.*log(hyp) + (1-y).*log(1-hyp));