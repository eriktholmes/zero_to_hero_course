# A small Generative Pre-trained Transformer (GPT)


This folder contains a build a miniature GPT model in PyTorch, following [Episode 3](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7) of Kaparthy's *Zero to Hero* series. It currently contains all the core pieces needed to scale up (though it remains a bit messy while I explore and iterate).

---


## Birdseye view:
We implement an **autoregressive decoder-only Transformer**, a type of model that predicts the next item in a sequence one step at a time, based only on past context. Here is a brief informal dictionary:
>- **Token** $\longrightarrow$ A basic unit of input/output in the model, like a character, word, or subword chunk.
>    - In this notebook, we use single characters (letters, numbers, punctuation), which makes things simple and helps us learn the fundamentals — though it requires longer sequences since each word is many tokens.
>    - More practical models use words or subword units, which lead to larger vocabularies but shorter sequences, and are more efficient for real-world tasks.
>- **Autoregressive** $\longrightarrow$ Predicts each next token based only on the previous tokens.
>- **Decoder-only** $\longrightarrow$ At each step, the model only has access to earlier positions — no peeking ahead.
>- **Transformer** $\longrightarrow$ A neural network architecture that uses **attention** to decide which previous tokens are most relevant at each step.
>- **[Attention](https://arxiv.org/abs/1706.03762)** $\longrightarrow$ (this ones fun) A mechanism that allows tokes to communicate, assigning relative importance to earlier tokens and using that information to refine the next-token prediction.

So in plain terms: this model learns to predict the next character in a sequence by looking back at all prior characters, assigning importance via attention, and generating probabilities for what comes next.

We then sample from these probabilities to produce text.

---

### 🔍 What kind of output can we expect from a model trained on Robert Frost's book of poems?

---

#### 0. First, a look at the training data.
For reference, we output the first poem in Frost's book of poems. This will serve as our reference in the models outputs below. Eventually, scaling up and running on a GPU we see it start to match the format and output "reasonable" text! 
```
THE PASTURE 


I'm going out to clean the pasture spring
I'll only stop to rake the leaves away
And wait to watch the water clear, I may
I shan't be gone long
You come too
I'm going out to fetch the little calf
That's standing by the mother
It's so young
It totters when she licks it with her tongue
I shan't be gone long
You come too

[1] 
```

#### 1. Let's first look at what the model produces before any training...
Here is the (random) generation based on the *Robert Frost* vocabulary
```
,Re]?{>;L-b[3z[Izh'[UOIl£“-pvw,¬*fc6-j&BKh■JGCz— c”rDfTh5y69|3!pU‘q
KxuzXUjdvx\pN1lU|}Im(8m~Qh£)Zj!5CK‘&&deR?KAv/LEf”O9rs:y5<IIH.db]0K qtif£F<ABM;uWB?—mz/{■(K5b<[¬*P“ltxV’’
]YjaFE2//jtX1a(“L]B)k24v[,N.>4B*f9¬;7777jfYI*kh’8!;)Zf^( R6qkUX?—u1>"VA;77~Z"n-bJmx
Q
P6:s"riHo3k„NLa]O[lPqpFwJpw<66G78^1|DqlQ,BQQp!vg’EAJG"‘—lx0z"R[SJVUX3|("0 c|zJssA
p4e.(0|{q|;¬X<G2■u1?eKAy^&E;X’¬u¬qHLf')RD36Mq
s/?~~’’8;)O!}p„W„’0jdHQc"Y{>7OO6H"|■I)Z5M/—9VZdYnO.6’*>‘”PvBLfDNtsK!1>AM6G
pfp'QZ B“:- h&z"1q"kn?‘“aA„U\0|4O3!qnX
```
#### 2. A simple bigram model (from `makemore`)
Next we implement a simple bigram model which is model that bases its next character prediction based ONLY on the current character (hence BI-gram). Note that even though this is extremely basic it shows a remarkable improvement to the untrained model above. What we see is that letters dominate (as one would expect!) but it also starts to mirror the format of the `Frosty` text file:
```
Towe ren trdishe ckend 
Ttoupat umid tw watharhey 1 me y w avoowoutonsl I fun t adone HT tee 
‘Wininond. t t. d acisek-ve hind beruis lagrsoineserso our pes ck. wenyopille arurourr 

T 
Tolinopro 

Choue we tr hew e the’Th t ge bownd s womileat any I camoouthed he 


'the tharfo s g 'th bathetr alenthed warophes 
The mery ner 

Win on 15 otlkerev, ubexe 



'toed It theld mithe tivey ak P tavere kind e offt o t t at 
Tincama f GR thathe s 

AR s to foun hary ake th! I’serod d 






[ the bisas.
```
#### 3. All the bells and whistles (on CPU)
We implement a transformer model with attention blocks, layernorm, dropout. Even with a simple CPU experiment (i.e. small parameters) it achieved much better results. 

```
Mal-hey neot. 

Beyt sost havaded yow I 
Jacone anly wer lowe t of worthe douth wit’s il 
I che. 

A to es coid bele there onen ars 
The won of tio has waith her agide 

Whin in ber were stem, shas thidg steerred hou chertick the tumee and him he, 
Wicht wich braind of frod, a and wack woonelt ef agu spolk, dor waie, 

Gound the innd., 

That ough lot 
Whathend Wherved, 
And culd. 6250LTh 

MAShe triese and abe fringevet chis hatirs dand. it go nee lack rel not in in wheny mare, gar do devotid. 
```
#### 4. The Finale... ported to GPU
To test the limits of our small GPT model, we scale up several core parameters — increasing the model’s depth, width, context length, and batch size — while training on a free-tier T4 GPU using Google Colab.

> - batch_size = 64
> - block_size = 128
> - learning_rate = 3e-4
> - n_embed = 256
> - n_heads = 4
> - n_layer = 4
> - dropout = 0.2

- Total training steps: 5000
- Runtime: ~9 minutes on a T4 GPU (Google Colab free-tier)

And here is some sample output (not perfect english... but another marked improvement): 
```
And in Polisely 
Of the , dared always to things with-sIngs, 
’He give used I neither and came— 

To learned me there with other they forget—in verning. 
You see and not blown?’ have to losen. 

God their race are that of those provals across the brook 
Be there blacked to himself on a man, 
But almost There thoughow hear were and fool. 

As much flowers, by a sepear things blomg. 
Less the whole of children to reason. 

The brittor every groutsion. Out keeped the bott 
Perchancheet the burieves
```
One more for good measure, and to highlight that the model is learning the structure of the Robert Frost book of poems:
```
ONENT 


And flower what checks a smile darled out end 
dows for a farmica cellar to fice. 

The fame needn’t know that’s have left. 


Neither expered. His earn one. 


[ 275 ] 
```






### Data: 
This notebook is designed to be flexible — I experimented with a few different .txt files for training:

- `RobertFrost_Poems.txt` – stylized poetry with strong rhythm and line breaks
- `Dwight.txt` – Dwight's dialogue from The Office
- `Quote_bank.txt` – general quotes and aphorisms
- `yoda.txt` – Yoda-isms
- `Shakespeare.txt` – classic and often used in GPT demos (this is what Kaparthy uses)
Each file is preprocessed into a character-level vocabulary. You can easily swap files and retrain to see how the output shifts.

