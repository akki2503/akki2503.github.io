# Review: Attention is All you Need

Here's the table of contents:

1. TOC
{:toc}

## WHy?
To computer representations of input and output - first of a kind attempt where a model relies entirely on self-attention wihtout using sequence-aligned RNN's or convolutions.
To enable more parallelization and less time to train with superior performance. 

## How?
Using self attention mechanism.
Most neural translation models follow encoder decoder scheme.
Transformers follow the same architecture using stacked self-attention and point wise fully connected layers for encoder and decoder.

- Encoder 
	Stack of 6 identical layers.
	each layer has 2 sub layers -  a) Multi head self attention mechanism
				 	b) Position wise fully connected feed forward network
	output of each sub layer is LayerNorm(x + sublayer(x))
	output dimension of each layer is 512 to facilitate residual connections

- Decode 
	Also consists of 6 identical layers
	It has 3 sub layers, an extra multi head attention layer for output of encoder stack
	have residual connections for sublayers and layer normalizations as encoder
	masking* ensures that predictions for position i depend only on known outputs at position less than i
	
- Attention 
	
	

## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![](/images/logo.png "fast.ai's logo")

## Code

General preformatted text:

    # Do a thing
    do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

    2

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Footnotes

[^1]: This is the footnote.

