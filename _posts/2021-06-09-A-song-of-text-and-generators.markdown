---
layout: post
title:  "A Song of Text and Generators"
date:   2021-06-09 14:16:52 -0400
category: project
img: /img/got.jpg
excerpt: "Training GPT-2 on Game of Thrones (Book 1) to generate new text for impatient fans of the series."
github: "https://github.com/rasim321/A-Song-of-Text-and-Generators"
---

A text generator for the Song of Ice and Fire (if GRRM won't finish his books, we must take it upon ourselves to do so)

Training GPT-2 on the first book of a Song of Ice and Fire to generate new text for impatient fans of the series.

**Example prompt: "A Lannister always"**

>Generated Text: "A Lannister always pays his debts, not for his own time… to the lord or the woman.” Varys sighed as Catelyn smiled, slipping uncertainly from her chair. “What if Gregor didn’t ask? Damn him. Just as treason falls.” “Tell him he was lying, to the council or the king?” Cersei rose. “There’s more to that than show, if it’s true. Lysa will send the queen back to Dragonstone, her own womb in her womb and keep the baby at her side. She wants to see her father, to see his joy, to see his company, to see Marillion the warm courtesies who are hard-pressed to keep them.” “The truth is, little sister,” Varys said. “If truth be told, then truth be told, the Lord Commander"

**Example Prompt: "Jon Snow"**

>Generated Text: Jon Snow?” Jon saw the king’s face, blind and speechless. He had never seen a man so beautiful. “I’ll send him back to the Eyrie,” Jon told him. “If I had, I’d sooner send them to Winterfell instead.” Ned would send word to his brother and to Winterfell and all the realm at once. He hoped he could ride out to Winterfell to face his father, not this one. He hoped he would remind him that he was not a man to be afraid. But it was not often that the king looked up to him with a glance at the Seven Kingdoms. It was seldom that when he saw the faces of men like Snow, he thought. He saw the faces of men like his own, men like old men who looked up to him in desperate hope. The dragons did not wait for him. He saw the faces of men"

**Example prompt: "Cersei sat on the iron throne"**

>Generated Text: Cersei sat on the iron throne, silent and melancholy, drinking wine from a cup he had left for the khalasar. “Lady Stark, I am pleased to know that you have been found.” “Very well,” Ned said. “I shall not forget it,” Catelyn said. She was seated on the iron throne, surrounded by her husband’s knights. “We are not to leave this chamber empty-handed, you have vowed to my love and honor.” “Is that understood, then?” Ser Rodrik said. “I would not betray your king, Catelyn,” she said. Catelyn had never loved a king before. She had always loved the khalasar, she knew it, and she knew it from her own heart. It was hard to lie so alone. “The khalasar is a sacred place."

# How to train GPT-2 on any text

## 1. Load the Packages

First, let's load the necessary packages. We are going to use the HuggingFace package to load the GPT-2 pre-trained model. Transformer models like GPT-2 are large and very expensive to train, so for our purposes, we can use pre-trained models (with their weights already trained on a large text corpus) and optimize on our target text. 


    import re
    import matplotlib.pyplot as plt
    import numpy as np

    !pip install transformers
    from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
    from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.utils.layer_utils import count_params


# 2. Read the Data

Although we will only use the first book, with enough time or compute, we can train the model on all five available books.


    #Read the books
    books =[]
    for book in ['got1.txt']: #we can also train on  ['got2.txt', 'got3.txt', 'got4.txt', 'got5.txt']
    book = "/content/" + book
    with open(book) as file:
        books.append(file.read())

    #Clean the books of "\n" tags
    books_cl = []
    for book_text in books:
    books_cl.append(book_text.replace('\n', ' '))

We do not want to take the punctuation and stop words from the text corpus because we want our language model to learn the structure of the text. It is a good idea to remove the table of content page and the general publisher text from the beginning of the book. Since we're using only one book, it's fine not to do so. 

# 3. Let's get our tokenizer

Tokenization is breaking down our text corpuses into smaller units. For example, if our text is "You know nothing Jon Snow", tokenizing this text will create ['You', 'know', 'nothing', 'Jon', 'Snow']. We have tokenzied this sentence at the word level, but we could have just as easily taken it at the character or sub-word level. Tokens are the most common unit of analysis for natural language processing. 

    gpt_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

Let's check the length of the first book 

    book_1 = books_cl[0]
    print(len(book_1.split()))

Now, we will tokenize the book with the max_length argument being slightly larger than the length of the book. 

    corpus_tk = gpt_tokenizer.encode(book_1, max_length=300000, truncation=True)

This may take a while. Once the tokenizer runs, it is a good idea to save the tokenized text. We can save it as an numpy file. 


    # Save the embeddings
    corpus_tk_arr = np.array(corpus_tk)
    np.save('book_1_embeddings.npy', corpus_tk_arr)

# 4. Pre-processing

Next, we create the traning and label data from our text corpus. For generating text, how this works is very interesting. We want the model to be able to accurately guess what the next words should be given a sequence of words. Therefore, the class label for this problem is the same text shifted by one word. We split the book into chunks of text first and then misalign the chunks by one word for each chunk. 

# Split into training chunks
training_chunks = []
block_size = 100

    # Simple algorithm to take the list of words and put them into chunks of text in a list
    for i in range(0,len(corpus_tk),block_size):
            training_chunks.append(corpus_tk[i:i + block_size])

    # Generate inputs and labels by offsetting the text by one word
    inputs = []
    labels = []
    for ex in training_chunks:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    print("inputs length:",len(inputs))
    print("labels length:",len(labels))

After creating the inputs and labels, we shuffle, batch, and prefetch the data in that order. The Autotune argument allocates the cpu budget for the data pipelines dynamically across all parameters. 


    # Autotune
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Batch Size and Buffer Size
    BATCH_SIZE = 12
    TRAIN_SHUFFLE_BUFFER_SIZE = len(inputs)

    # Create TF Dataset
    train_data = tf.data.Dataset.from_tensor_slices((inputs, labels))

    #############
    # Train data
    #############
    train_data = train_data.shuffle(buffer_size=TRAIN_SHUFFLE_BUFFER_SIZE)
    train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)
    train_data = train_data.prefetch(buffer_size=AUTOTUNE)

    print("train_data",train_data)

# 5. Build the Model

We initiate the pre-trained weights for the Distil GPT-2 model from Huggingface. 

    # Build the model
    model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")

Distil GPT-2 is the most light weight version of GPT-2, hosting 84M parameters compared to the 124M for the full GPT-2 model. We use this model since it is twice as fast as the full GPT-2. 

# 6. Hyperparameter Optimization

Next, we will set some hyperparameters to ensure that our model runs efficiently. The learning rate determines how much the weights are updated in each epoch during training. We introduce a learning rate that is not so low that it causes a vanishing gradient problem and not so high that the global minima is never reached and the model converges to a suboptimal solution.

    #Hyperparamters
    learning_rate = 3e-5 

The epsilon is a small constant that prevents dividing by zero in the denominator. The higher the epsilon the smaller the weight updates. 

    epsilon = 1e-08

Cliptnorm prevents the exploding gradient problem. This is called gradient norm scaling where if the vector norm of a gradient exceeds a threshold (in our case 1.0) then the values of the vector will be rescaled so that the norm equals the threshold.

    clipnorm = 1.0

An epoch is one forward pass and one backward pass of the entire dataset through the model. 

    epochs = 30

Now we can print the model summary and initialize the model optimizer with our chosen hyperparameters. 

    # Print the model architecture
    print(model.summary())

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)

The loss function will be sparse categorical crossentropy since our labels are mutually exclusive and the metric will be accuracy. 

    # Loss
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #metric
    metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

# 6. Compile and Train

Now we just compile and train our model. 

    # Compile
    model.compile(loss=[loss, *[None] * model.config.n_layer],
                    optimizer=optimizer,
                    metrics=[metric])

    # Train model
    start_time = time.time()
    training_results = model.fit(
            train_data, # train_data.take(1000) for testing
            epochs=epochs, 
            verbose=1)
    execution_time = (time.time() - start_time)/60.0
    print("Training execution time (mins)",execution_time)

# 7. Save Model and Weights

It's a good idea to save our model and the weights once we are done with training. We can then just load the trained model whenever we want to generate new text. 


    # Save the weights
    model.save_weights('gpt_got1')

    # And the model
    model.save('my_model')

# 8. The Winds of Winter

Now let's generate our text!

    # The model will use this text to generate new text
    input_text = "Cersei sat on the iron throne"

    # Tokenize Input
    input_ids = gpt_tokenizer.encode(input_text, return_tensors='tf')
    print("input_ids",input_ids)

    # Generate outout
    outputs = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=200, 
        top_p=0.75, 
        top_k=0
    )

    print("Generated text:")
    display(gpt_tokenizer.decode(outputs[0], skip_special_tokens=True))

And there we have it! Our text generator is live. 