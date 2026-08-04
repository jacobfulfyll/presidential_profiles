# Coverage-pressure analysis preregistration (v1)

Frozen before the confirmatory build. The hypothesis is that modern presidents
cover more detailed topics per major annual speech while sustaining each topic
for fewer consecutive words. The primary sample is State of the Union/annual
messages with at least 12 paragraphs. It compares the pooled 1860, 1890, and
1920 eras with 1950, 1980, and 2010.

Topic exposure splits each paragraph's word count equally across its normalized
AI topic labels. Breadth is exp(Shannon entropy) of speech-level exposure.
Depth is the length-biased episode size, sum(run words squared) / sum(run
words), where an episode is a consecutive paragraph run carrying one topic.

Presidents and then their speeches are resampled in a 20,000-draw hierarchical
bootstrap. The two one-sided primary tests are Holm-adjusted as one family.
Coverage pressure is supported only when adjusted breadth is positive and
adjusted depth is negative at alpha .05. All-speech raw, genre-standardized,
and length/paragraph/medium-controlled estimates are sensitivity arms. Leading
topic share, median episode length, singleton runs, adjacent-paragraph Jaccard,
and exposure in episodes of at least 250/500 words are secondary outcomes.

All arms, exclusions, sample sizes, and null results will be published. P-values
are frequentist no-change surprise rates, never probabilities that a finding is
random or real.
