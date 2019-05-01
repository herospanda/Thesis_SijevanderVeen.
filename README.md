Analyzing Facebook posts and comments in French and Arabic, using a Recurrent Neural Network and Topic Modeling

1.  INTRODUCTION

The idea for this project originated by Radio Netherlands
worldwide media (RNW). RNW creates communities online
to contribute to social change. The topics they cover reflect
young  peoples  needs  in  relation  to  love,  relationships  and
their hopes and ambitions for their societies. Local teams of
media-makers manage the digital communities. In turn, local
teams  build  and  coordinate  networks  of  young  people  who
produce content to engage the wider community across our
thematic areas [2].
This research can contribute to retrieve more and in depth
information from the content on the discussion platforms. In
this research will be worked with French and Arabic data, the
current  data  method  which  is  used  to  store  and  explore  the
data already provides information, like sentimental and emo-
tional information. This information is based on predictions,
by observing the data manually it is easy to observe that the
model used by the Chemistry makes mistakes. Therefore in
this study a new model is applied on the data. Besides a better
predicting method for emotional and sentiment information,
also topic modeling is applied on the data. By applying topic
models  on  these  discussion  data,  it  would  be  interesting  to
see  if  topic  modeling  is  a  suitable  methods  to  distinguish
between comments, how likely they are with the post.
For this study the research question is defined as followed:
Comparing  Facebook  discussions  in  French  and  Arabic  to
retrieve  in  depth  information,  using  Recurrent  Neural  Net-
work  and  Topic  Modeling.  To  split  the  question  in  pieces,
the following sub questions are defined:
1)  How does the current prediction method for sentiment
and emotional information performs?
2)  Will probabilities for each label per word optimize the
results?
3)  Will  using  a  Recurrent  Neural  network  optimize  the
results?
4)  Can  Facebook  conversations  be  compared,  using  the
the likeliness of the post as gold standard?
5)  Is topic modeling a good measurement for likeness of
post and comments?

II.  RELATED WORK

A. Stems and Phrases

Before a text data can be used for language model, clean-
ing  methods  can  help  to  optimize  the  performance.  Words
can have different morphological variations as in inflectional
(plurals,  tenses)  or  derivational  (making  words  nouns).  In
general  they  have  the  same  meaning  or  almost  the  same
meaning. Stemmers can be used to reduce the morphological
variations  of  similar  words  to  a  common  stem.  In  many
causes  suffixes  will  be  removed.  The  effect  of  stemming  is
general small but significant, which can be in some languages
crucial. There are two ways to perform stemming; by using a
list of related words (dictionary) or using an program which
determines  words  (algorithm).  Algorithms  can  make  easily
mistakes, for example with by removing and assuming it is
plural. In cause of supplies to supplie is a false negative and
ups to up results in a false positive.
Phrases are more precise than single words. For example
documents  containing  black  sea  vs.  two  words  black  and
sea. And ambiguous for example big apple vs apple. Phrases
can be recognized in two different approaches, by identifing
syntactic  phrases  using  a  part-speech  (POS)  tagger  or  by
using  word  n-grams.  a  simple  data-driven  approach,  where
phrases are formed based on the unigram and bigram counts.
The bigrams with scores above the chosen threshold
are then used as phrases. POS taggers use statistical models
of  text  to  predict  syntactic  tags  of  words  These  tags  can
be  used  to  find  phrases  in  textual  data.  POS  tagging  is
too  slow  for  large  collections.  Frequent  n-grams  are  more
likely to be meaningful phrases. Language models are used
on  many  languages,  mainly  frequently  spoken  languages,
and  for  many  purposes.  To  estimate  the  relative  likelihood
of  different  phrases  is  useful  in  many  natural  language
processing  models.  Languages  models  are  used  for  speech
recognition, part-of-speech-tagging, handwriting recognition
and  many  other  applications.  Bag  of  words  representation/
unigram are commonly used for query likelihood models. A
separate language model is associated with all documents in a
collection. The documents are ranked with the probability of
a given query in a language model. A major problem in build-
ing a language model is the sparsity of data. In the training of
the model not all words are observed, which results in zero
probabilities.  This  can  be  solved  by  smoothing  techniques,
assuming that a word is depending on the previous word(s)
in a sentence (n-gram model). Words in bigram and trigram
denotes for a language model with n = 2 and n = 3.
Different  type  of  models  were  used  for  estimating  con-
tinuous  representation  of  words.  Two  well  known  models
are  Latent  Semantic  Analysis  (LSA)  and  Latent  Dirichlet
Allocation  (LDA).  LDA  is  a  probabilistic  model  that  treats
documents as mixtures of topics. It learns topics as discrete
distributions  (multinomials)  over  the  event  patterns,  and
thus  meets  our  needs  as  it  clusters  patterns  based  on  co-
occurrence  in  documents.  LDA  will  used  to  identify  topics
in the comments in this project.
The use of a neural network is more favored to analyze the
continuous representation of words. In previous research has
shown that the performance of a neural network was better
than  LSA.  LDA  can  become  computationally  expensive  on
larger data sets [6].

B. Neural networks

Neural  language  models  are  made  and  trained  as  prob-
abilistic  classifiers  that  are  taught  to  predict  probability
distribution over an vocabulary.
For textual classification are the common neural network
algorithms  are  used  like  stochastic  gradient,  forward  and
back propagation. One neural network type used for language
is the recurrent language model, which can predict a window
of previous or future words [14].
Instead  of  using  a  neural  net  to  predict  probabilities  a
distributed  representation  which  encodes  in  the  networks
hidden layers as representations of words can be used. Words
are mapped in a n-dimensional space (embedding). The char-
acteristic for embedding is that the semantic relationships are
linear combinations. 
Other  options  are  adjectives  base  form  vs.  comparative
(e.g.,  good,  better),  nouns  singular  vs.  plural  (e.g.,  year,
years) and verbs present tense vs. past tense (e.g., see, saw)
[2] [5]
In this project a recurrent neural network will be used to
make vectors of the Facebook topics and the comments. The
idea behind RNNs is to make use of sequential information.
In a traditional neural network we assume that all inputs (and
outputs)  are  independent  of  each  other.  But  for  many  tasks
thats a very bad idea. If you want to predict the next word in a
sentence you better know which words came before it. RNNs
are called recurrent because they perform the same task for
every element of a sequence, with the output being depended
on  the  previous  computations.  Another  way  to  think  about
RNNs is that they have a memory which captures information
about what has been calculated so far. In theory RNNs can
make use of information in arbitrarily long sequences, but in
practice they are limited to looking back only a few steps

III.  METHODOLOGY

A. DESCRIPTION OF DATA

In table I the values from the Facebook dataset from DRC
are shown. This dataset consits out French data collected over
2 years.

B. METHODS

1) How does the current prediction method for sentiment
and emotional information performs?:

For all languages the
dataset consist for each comment a polarity class (negative,
neutral and positive) and emotional class (e.a. neutral, hap-
piness, surprise, anger and sadness). Per dataset the number
of specificity of classes are different. These classes used as
labels for a machine learning task. The textual comment can
be  used  as  feature  (after  tokenizing  and  select  a  specific
number of features.). It would interesting to see which word
correlates  with  specific  sentiments  or  polarity.  Chemistry
the  data  converter  program  provides  information  about  the
sentiment and emotion. In table II are few examples shown
for French data.

2) Will probabilities for each label per word optimize the
results?:

The current predictions aren’t correct all the time.
n order t optimize the predictions results a gold standard is
searched. On Kaggle a data set is found which consists out
1100  words  in  English  with  a  probabilities  for  7  emotional
classes.  In  order  to  use  this  dataset  to  train  a  classifier  on,
the  are  translated  to  French  and  Arabic  using  the  Google
translate API in python.

3) Will using a Recurrent Neural network optimize the
results?:

Supervised  training  methods  like  Random  forest
and Naive Bayes classification are used to classify text.

4) Can Facebook conversations be compared, using the
the likeliness of the post as gold standard?:

Besides a better
predicting method for emotional and sentiment information,
topic  modeling  is  applied  on  the  data.  By  applying  topic
models  on  these  discussion  data,  it  would  be  interesting  to
see  if  topic  modeling  is  a  suitable  methods  to  distinguish
between  comments,  how  likely  they  are  with  the  post.  To
measure  the  performance  an  gold  standard  is  needed.  The
cosine  similarity  will  be  calculated  between  the  post  and
comment.

5) Is topic modeling a good measurement for likeness of
post and comments?:

Topic modeling is mainly used to com-
pare document of text with each other or documents with a
text query. Topic modeling is not used yet to compare smaller
pieces  of  text  like  sentences  in  a  Facebook  conversation.
After a LDA model will be trained if available conversational 
data. The LDA model will consists out 10 topics (For each
input, 10 output scores will be generated). On each topic and
his comments (separately) LDA will be applied. The output
scores from the comments can be compared to the scores of
the  post.  This  results  in  a  new  ranking  for  each  comment
to  their  post.  This  ranking  can  be  compared  to  the  ranking
from the gold standard.

REFERENCES

[1]  https://www.rnw.org/

[2]  Mikolov,  Tomas;  Chen,  Kai;  Corrado,  Greg;  Dean,  Jeffrey  (2013).
”Efficient estimation of word representations in vector space”

[3]  http://mt-class.org/jhu/slides/lecture-nn-lm.pdf

[4]  Nathanael  Chambers  and  Dan  Jurafsky,  Template-Based  Information
Extraction without the Templates

[5]  Mikolov, T., Yih, W.-T. and Zweig, G. Linguistic Regularities in Con-
tinuous Space Word Representations. HLT-NAACL, 746-751. 2013

[6]  Mikolov  T,  Sutsekever  I,  Chen  K,  Carrdo  G,  Dean  J.  Distributed
representations   of   words   and   phrases   and   their   compositionality.
NIPS’13 Proceedings of the 26th International Conference on Neural
Information Processing Systems - Volume 2, pages 3111-3119

[7]  Stevan   Ostrogonac,   Dragia   Mikovi,   Milan   Seujski,   Darko   Pekar
and  Vlado  Deli.  A  Language  Model  for  Highly  Inflective  Non-
Agglutinative Languages.

[8]  James  Allan  and  Giridhar  Kumaran.  Details  on  Stemming  in  the
Language Modeling Framework

[9]  https://medium.com/greyatom/learning-pos-tagging-chunking-in-nlp-
85f7f811a8cb

[10]  Michael Collins. Tagging Problems, and Hidden Markov Models.

[11]  Statistical  Language  Modeling  for  Automatic  Speech  Recognition  of
Agglutinative Languages.

[12]  Dictionairy for English-Rundi https://glosbe.com/en/rn/

[13]  Charlotte Vonkeman, Kyle Snyder. Yaga Burundi.

[14]  Mikolov T. Karafiat M. Burget L. Cernocky J. Khudanpur S. Recurrent
neural network based language model.

[15]  RNW media website: https://www.rnw.org/about-us/

[16]  Mijit  Ablimit,  Sardar  Parh,  Askar  Hamdulla,  Thomas  Fang  Zheng.
A  Multilingual  Language  Processing  Tool  for  Uyghur,  Kazak  and
Kirghiz.

[17]  Polyplot https://hub.packtpub.com/morphology-getting-our-feet-wet/

[18]  S.  Dai,  Q.  Diao  and  C.  Zhou  Performance  comparison  of  language
models for information retrieval.

[19]  Building   an   RNN   in   Tensorflow   with   Pretrained   Word   Vectors
http://www.brightideasinanalytics.com/rnn-pretrained-word-vector
