# Bigram implementation

This folder contains our implementation of the bigram (character-level) language model following [episode 2](https://youtu.be/PaCmpygFfXo?si=O-M-9zdDWNpGVGur) of Andrej Karpathy's *makemore* series. 

The goal is to build a model that will predict the next character in a string given the current character. We do this in two steps, first using a simple counting approach, then by building a simple neural network and training it on bigrams. We use the same name.txt dataset as Andrej.

---

## Part 1: naive counting
Following the outline in the video we first build a bigram model from scratch by counting bigram occurances in the dataset.
  - loop over names in the dataset and count occurances of bigrams.
  - store raw counts in a 27x27 tensor (N[i][j] contains the number of times that the 'jth' character follows the 'ith' character)
  - normalize this counts matrix to produce a probability distribution
  - using torch.multinomial we sample from the model and sample new 'names' from this distribution

---

## Part 2: Neural network
Next, we build a neural network to complete the same task:
  -  count bigrams as above to get the training pairs: input (current char) and target (next char)
  -  one-hot encoding for input data 
  -  initialize a simple, single layer, network
  -  apply softmax to convert the logits (log output) to probabilities
  -  train the model using negative log likelihood loss (in 100 epochs we achieve about the same loss as the counting approach above)
  -  sample from this model to produce new 'names'

After about 100 epochs, the network achieves a similar loss to the naive counting approach and generates similar looking names.


> ### Next steps
>  - Try to implement a trigram model using both a frequency based approach and the neural network
>  - experiment with hidden layers
>  - compare models in terms of performance and output quality

---

This notebook serves as a checkpoint in my ML learning journey - treated like a worksheet - aimed at understanding the building blocks of character level language models before scaling up to more complex models involving RNN/Transformers.
