 The files “spamiam.txt” and “saki story.txt” available in the current folder have poetry and prose of specific genres. For this problem, use the text in these files to empirically estimate probabilities and transition probabilities as indicated. Ignore any characters in these files other than the 26 alphabets ’a’-’z’ (use white space and carriage returns as indicated in specific
 parts). Also ignore any case distinctions among alphabets (example ’C’ and ’c’ are equivalent).
 For each of the sub-parts indicated below, print the 100 words that you generate in the form of
 a 10×10 array and circle any valid English words that you recognize.
 (a) Assuming that the 26 letters of the alphabet are equiprobable. Generate one hundred random 4
 letter words by selecting the 4 individual letters of each word independently.
 (b) Estimate the probabilities of individual letters using “spamiam.txt”. Generate one hundred random
 4 letter words by selecting the 4 individual letters of each word independently according to the
 estimated probability distribution.
 (c) Again using the file “spamiam.txt”, estimate the transition probabilities, P(xn+1|xn), for all 26
 possible values of xn- the nth letter in a word and xn+1- the (n + 1)th letter in a word (assume
 that these probabilities are independent of n). Also for this part and the next, for your estimation of
 transition probabilities, use only the letters inside a word for the computation and do not incorporate
 letters from adjacent words (with a blank in between). Generate one hundred random 4 letter words
 by first generating a letter at random according to the probability mass function (pmf) in 2b and
 then generating remaining letters according to appropriate transition probabilities. Note: You may
 default to the model of 2b if you end up with a situation where your estimate of P(xn+1|xn) is zero
 for all values of xn+1.
 (d) Once again use the file “spamiam.txt”, to estimate the transition probabilities P(xn+1|xn,xn−1),
 for all possible values of the successive letters. Generate one hundred random four letter words using
 these estimated probabilities. Make reasonable assumptions that generalize what was indicated
 in 2c.
 (e) Repeat parts 2b-2d using the file “saki story.txt”.
 (f) Comment on your results.
 (g) Extra Credit: Estimate the entropy rate for each of the Markov models you developed and compare
 these both across models for a single data file and across the two data files. You may need to make
 suitable assumptions in order to determine your answers (which may be hard/impossible to validate).

 generate a file to hold all the answers, and store the python code for each problem in seperate python files.