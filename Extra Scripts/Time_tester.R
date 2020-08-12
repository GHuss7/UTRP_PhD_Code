# Time tester

t1 <- Sys.time()

t2 <- Sys.time()

t2-t1

# How to calculate waiting time and line probabilities
f1 <- 1/12
f2 <- 1/30

f_comb <- f1 + f2

1/(f_comb)/2

# Line 1 prob
f1/f_comb

# Line 2 prob
f2/f_comb


waiting_time = f_comb/2

line_1_prob = f1/f_comb

line_2_prob = f2/f_comb