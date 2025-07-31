# Zero-to-hero-course
Work through of Karpathy's "zero to hero" course on Youtube

(first project on GitHub... working to transition into the ML/AI space and wanted to start simply with this video series)

**Part 1:** building micrograd.
-_ Key takeaway_: I love calculus! Another relevant point for students... why do we care about derivatives and local minima?? Machine learning! A better answer, perhaps, than 'to find the maximum area of a garden using a fixed amount of material...'





## Part 3: GPT Mini — “Frosty”

### What is it?

This is our mini (character level) GPT, modeled after Karpathy’s implementation. Think of it as a tiny version of ChatGPT — but without the *chatting* fine-tuning. In other words, it won’t answer questions, but it can generate text — specifically, poetry in the style of Robert Frost.

We call it **Frosty**, since we trained it on a collection of Robert Frost’s poems.

Let’s quickly break down what “GPT” means:
> - character level - as above, this means the dictionary of tokens is simply the individual letters/symbols in the text file. 
> - **G**enerative — the model can generate text, sampling tokens one by one  
> - **Pr**e-trained — it first learns statistical patterns from a dataset (here: Robert Frost's poetry)  
> - **T**ransformer — it’s built using the Transformer architecture from ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)  

So: **a small Transformer trained to continue Robert Frost’s poetic style**. You can think of it as a probabilistic “document completer” — a model that sees the first few words of a line and tries to finish it, having only been exposed to poems by Frost. 

Here’s a sample from the original dataset — one of my favorite poems, and the first I ever read from Frost:

```
STOPPING BY WOODS ON A 
SNOWY EVENING 

Whose woods these are I think I know. 
His house is in the village though; 

He will not see me stopping here 
To watch his woods fill up with snow. 

My little horse must think it queer 
To stop without a farmhouse near 
Between the woods and frozen lake 
The darkest evening of the year. 

He gives his harness bells a shake 
To ask if there is some mistake. 

The only other sound’s the sweep 
Of easy wind and downy flake. 

The woods are lovely, dark and deep. 
But I have promises to keep, 

And miles to go before I sleep, 

And miles to go before I sleep. 


[275] 
```
So, this is what we are aiming for... let's see what/how we do. 






---
---

Here’s how we start:







### And... live from Frosty
```
ONENT 


And flower what checks a smile darled out end 
dows for a farmica cellar to fice. 

The fame needn’t know that’s have left. 


Neither expered. His earn one. 


[ 275 ]
```
