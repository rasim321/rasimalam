---
layout: post
title:  "Book Recommender"
date:   2021-06-10 16:16:52 -0400
category: project
img: /img/bookrec.jpg
excerpt: "Book recommender that uses plot summaries from GoodReads and embeddings from pre-trained transformers to recommend books."
github: "https://github.com/rasim321/Book_Recommender"
---

<img src="/img/bookrec.jpg" />

Check out the Book Recommender at [gr-bookrec.herokuapp.com](https://gr-bookrec.herokuapp.com) 

The Book Recommender uses metadata scraped from [GoodReads](https://www.goodreads.com) to recommend interesting and relevant books. 

**How to use:** The easiest way to use it is at the web app above. The database is updated regularly so if you cannot find your recommendations now, be sure to check back in 24 hours. You can also run it on google colab by scraping any GoodReads book list and run your query at the bottom of the notebook. The recommender can also take in book names to build the dataset.

The web scraper collects title, author, reviews, ratings, plot summaries, and tags data to create the website's database. Goodreads is a wealth of information on books but not very accessible in an analyzable format. For example, a typical booklist looks like this:

![image](https://user-images.githubusercontent.com/59543579/123773613-c95fe400-d89a-11eb-95b0-855072560a39.png)

And although the information is available to create a recommendetion algorithm, the dataset would be better served in a tabular format. To create the recommender, I first built a web-scraping tool to download Goodreads lists or a list of book titles into a dataframe. Here's the same book list scraped and transformed into a tabular format:

![image](https://user-images.githubusercontent.com/59543579/123776724-8c492100-d89d-11eb-9159-4493e12cae7d.png)

This is the first step to building the recommender.

# Part 1 - GoodReads Web-Scraper

In this section, I will go over how the web scraper works and some of the challenges in creating a website specific scraper.  Websites generally have a EULA or a robots.txt file that specify the level of access bots have in the website. It is important to read these files to know what is and isn't allowed. 

From the robots.txt file for Goodreads, we find that booklists are not disallowed. 

First, let's get the libraries required to build our web-scraper. 

    import argparse
    from datetime import datetime
    import json
    import os
    import re
    import time
    from urllib.request import urlopen
    from urllib.error import HTTPError
    import bs4
    from googlesearch import search 
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')

The web scraper is built with several functions, each of which target a particular group of data. The **Rater** function below fetches the average rating and the total number of ratings for a book. 

    def Rater(ratings):
        """
            Takes the ratings input from GoodReads and returns rating average and 
            number of ratings

            :param p1: Ratings input
            :return: average rating and number of ratings
        """ 
        #Lists for Average Rating and Number of Reviews
        rating_avg= []
        rating_num = []

        for i in ratings: 
            rating_avg.append(re.findall("\d+\.\d+", i)[0])
            try: 
                rating_num.append(re.findall("(?<=—)(.*)(?=ratings)",i)[0])
            except IndexError:
                rating_num.append('0')

        rating_avg = [float(i) for i in rating_avg]
        rating_num = [int(j.replace(",", "").strip()) for j in rating_num]

        return rating_avg, rating_num

Regex works very well to find specific patterns in HTML. In this case, we are using regex to find the average rating, which is in the format ("number", ".", "number") To make sure our code works well at scale, we have to make sure that edge cases and errors are accounted for. For example, if the number of reviews for a book (let's say a brand new not-so-well-known book) was zero, the web-scraper will account for the "IndexError" andjust return the number 0 at rating_num.

Similarly, the **GetDetails** function that collects the plot summary, tags, and number of reviews from a book's page. Notice that we put in intentional time pauses with `time.sleep(1)` of one second to make sure that we are not requesting access from the server too often. 

    def GetDetails(links):
    """
        Visits each book link and returns the plot summary, tags, and number of
        reviews.

        :param p1: GoodReads url of books
        :return: plot summary, tags, and number of reviews
    """ 

    url = 'https://www.goodreads.com/'
    plot_df = []
    book_all_tags = []
    n_reviews = []
    count = 0

    for link in links:

        book_page = urlopen(url + link)
        soup = bs4.BeautifulSoup(book_page, "html.parser")
        count += 1

        plots = soup.select('#description span')
        if len(plots) > 1:
            summary = []

            for plot in plots:
                summary.append(plot.text)
            
            plot_df.append(summary[1])
            
        else:
            for plot in plots:
                plot_df.append(plot.text)


        #Get book tags
        book_tags = []

        tags = soup.select('.bookPageGenreLink')
        for tag in tags:
            book_tags.append(tag.text)

        book_tags = [x for x in book_tags if not any(c.isdigit() for c in x)]
        book_all_tags.append(book_tags)
        time.sleep(1)
        print("Book: " + str(count) +  " out of " + str(len(links)))


        #Get Number of Reviews:
        reviews = soup.select('.reviewControls--left.greyText')
        rev = []
        for review in reviews:
            rev.append(review.text)
        rev = re.findall("\s\d+\s|\d+,\d+", rev[0])
        rev = [item.replace(",", "") for item in rev]
        n_reviews.append(int(rev[1]))
        
    return plot_df, book_all_tags, n_reviews

It's a good idea to interact with the web page that we are scraping and notice the idiosyncracies that govern the data that is displayed. For example, in the GoodReads site, the book pages have two kinds of plot summaries. If the plot summary is short, it just appears on the web page. However, if it is slightly longer, the user must click a "...more" button to then access the full plot summary. How do we tell our web scraper to check for this option? 

We need to find the class id that tells us whether the webpage hosts two plot summaries one short and one long. Now, we can check the html to find these tags manually, however there is an easier process to do this. I use a browser extension [SelectorGadget](https://chrome.google.com/webstore/detail/selectorgadget/mhjhnkcfbdhnjickkkdbjoemdmbfginb?hl=en) to easily locate the CSS selector. 

![image](https://user-images.githubusercontent.com/59543579/123855015-677c9a00-d8ed-11eb-9bc2-2c2d3c4edbc3.png)

Notice that for tags and reviews as well, we have to clean the data internally within the function to get the necessary data point from the book webapge. This requires some simple string manipulation using regex, and if necessary some list comprehension to get the desired data. 

# Part 2 - Adding Useful Features to the Web-Scraper

When downloading larger datasets, with hundreds or even thousands of books, we want to make sure that if the program gets interrupted we won't lose progress on the scraping. This could happen due to faulty internet connections, server timeouts, or extreme edge cases that break our code. For these cases, we will build several features in the web-scraper that allows checkpointing, pre-defined number of downloads, downloading books from the middle of a list, and iterative saving. We will also handle anticipated errors. 

**Features:**
1. An argument for the number of books we want to download 
2. A checkpointing system so that if something goes wrong we do not lose the whole dataframe
3. A way to save the files intermittently (for example, after 50 books)
4. If program gets halted for some reason, it can restart from a particular check point
5. If arrays are of different size, handle errors

Now that we have defined our features, let's build this function.

    def ListSaver(link, checkpoint_n = 50, save_csv = False, start_n = 0, num_books=1000000):
    """
        ListScraper takes a list of books, gets the author and book names, and the
        individual links to the book's Goodreads pages. It passes the link to Rater
        and GetDetails functions to receive additional data on each book. It returns
        a dataframe of all the above information for all books in the list.

        :param p1: Link to GoodReads book list
        :param p2: number of books to save at each checkpoint, by default = 50
        :param p3: Boolean to save a csv file at each checkpoint
        :param p4: start at 0 by default, but option to start at some other point
        :param p5: number of total books to save
        :return: Dataframe with book details: title, author, avg rating, number
        of ratings, link, plot summary, tags, and numnber of reviews.
    """ 
    #Check that the checkpoints are smaller than num_books
    assert checkpoint_n < num_books, "checkpoints must be smaller than number of books requested"

Let's make it so that our function can handle both full urls, and abbreviated book urls.

    url = 'https://www.goodreads.com/'

    #Check if the link contains a Goodreads list
    if "list" in link:
        pass
    else:
        print("This link does not contain a GoodReads list")

    #Strip the list url from the full url
    if link.startswith(url):
        list_url = link.replace(url, '')
    else:
        list_url = link

Then, we want to create lists for the data we are interested in. We are also initializing a count for the number of books that will be downloaded. 

    #Make lists for the data we want
    titles = []
    authors = []
    ratings = []
    book_links = []
    n = 0 #no. of books

Now, list pages on GoodReads have 100 books on each page. Our function will first iterate through the pages and collect the book titles, authors, and links first.  

    #Loop through the number of pages
    while (list_url != None):

        #Set up the URL
        source = urlopen(url + list_url)
        soup = bs4.BeautifulSoup(source, "html.parser")

        #Get all titles from this page
        books = soup.select(".bookTitle span")
        for book in books:
            titles.append(book.text)
            n += 1
        
        #Get the authors
        writers = soup.select(".authorName span")
        for author in writers:
            authors.append(author.text)

        #Get the ratings:
        rates = soup.select(".minirating")
        for rate in rates:
            ratings.append(rate.text)

        #Get the links:
        links = soup.findAll('a', {'class': 'bookTitle'})
        for link in links:
            book_links.append(link['href'])

        #Check if there is a next page and either end the program or continue to next page
        if not soup.select(".pagination"):
            list_url = None
        else:
            next = soup.select(".pagination")[0].select(".next_page")[0].get('href')
            if next != None:
                list_url = next
                print("Going to next page...")
            else:
                list_url = next
                print("No more pages to scan.")
    
We introduced a large default number of books argument in the function but for most cases, the number of books we want would be much less than that. We collect the number of books in the list and pass that value as the new value for the number of books argument. 

    #If num_books is larger than the total no. of books in the list
    #change num_books to the size of the list n.
    if num_books > n: 
        num_books = n
    else: 
        pass
    
Next, we create an empty dataframe where all of the data collected will be stored. 

    book_master = pd.DataFrame({
        'title': [],
            'author': [],
            "avg_rate": [],
            "number_of_ratings": [],
            "link": [],
            "plot_summary": [],
            "tags": [],
            "reviews": []
            })
    
this is where we will introduce checkpointing in our function. The number of files that will be saved can be found by dividing the number of total books downloaded by the checkpoint number. The remainder of the books must be handled separately after the saved files have been completed. We will get to this shortly. We then create a moving "start n" and "end_n" to keep track of the book titles, authors, and links from our original list where all the book data resides. For the rest of the data, ratings, reviews, tags, and plot summaries, we use our helper functions **Rater** and **GetDetails**.

    #Number of Saves
    save_n = num_books//checkpoint_n
    #Remainder books after the last checkpoint
    save_rem = num_books % checkpoint_n
    #Start_n and End_n
    start = start_n
    end = start + checkpoint_n

    while end <= num_books:

        #Transform string ratings to average and number of ratings
        rating_avg, rating_num = Rater(ratings[start:end])
        print(len(rating_avg), len(rating_num))


        #Get book details: plot summary, tags, and number of reviews
        plot_df, book_all_tags, num_reviews = GetDetails(book_links[start:end])
        print(len(plot_df), len(book_all_tags), len(num_reviews))

        book_dict = {
            'title': titles[start:end],
            'author': authors[start:end],
            "avg_rate": rating_avg,
            "number_of_ratings": rating_num,
            "link": book_links[start:end],
            "plot_summary": plot_df,
            "tags": book_all_tags,
            "reviews": num_reviews
        }

        book_df = pd.DataFrame({key:pd.Series(value) for key, value in book_dict.items()})

        start += checkpoint_n
        end += checkpoint_n
        book_master = book_master.append(book_df, ignore_index = True)
        yield book_master

        #Save
        if save_csv:
        book_master.to_csv(r'books-' + str(start) + '.csv', index=False, header=True)

This loop breaks when the end_n supersedes the number of books to be downloaded. At this point, we must download the remainder of the books in the list. This results in a neat dataframe with all the relevant data from the GoodReads list. And any break in the code still leaves us with all the book data saved till the last checkpoint. To make sure we do not lose too much data if that happens, we can choose a small checkpoint number like 10.

    #Last chunk of books after last savepoint
    
    #Transform string ratings to average and number of ratings
    rating_avg, rating_num = Rater(ratings[start:num_books])
    print(len(rating_avg), len(rating_num))

    #Get book details: plot summary, tags, and number of reviews
    plot_df, book_all_tags, num_reviews = GetDetails(book_links[start:num_books])
    print(len(plot_df), len(book_all_tags), len(num_reviews))
    
    book_end = {
        'title': titles[start:num_books],
        'author': authors[start:num_books-1],
        "avg_rate": rating_avg,
        "number_of_ratings": rating_num,
        "link": book_links[start:num_books],
        "plot_summary": plot_df,
        "tags": book_all_tags,
        "reviews": num_reviews
    }

    book_df = pd.DataFrame({key:pd.Series(value) for key, value in book_end.items()})

    start += checkpoint_n
    end += checkpoint_n
    book_master = book_master.append(book_df, ignore_index = True)
    yield book_master
    
    #Save 
    if save_csv:
        book_master.to_csv(r'books-' + str(num_books) + '.csv', index=False, header=True)
    
    return book_master

Now that we have our dataset, we have to write one more function to check for missing data and books in other languages. We determine whether a book is in English by removing books where a large number of words in the plot summary (over 40 percent) is in another language. It's a good idea to clean the dataset at this stage, and not in the scraping stage because we want to encumber the server as little as possible. We will call the new function **BookCleaner**. The function returns a cleaned dataframe as well as the abstracts, which is our primary data to build the recommendation algorithm. 

    def Book_Cleaner(book_df):

    """
    Removes missing plot summaries and non-English books from the
    book dataframe

    :param p1: Book Dataframe
    :return: Return cleaned dataframe and plot_summaries
    """

    ## Remove rows with missing plot_summaries

    #Extract the abstracts
    abstracts = book_df['plot_summary']

    #Index of all the complete cases
    abs_null = abstracts.notnull()

    #Dataframe with complete plot_summaries
    book_complete = book_df[abs_null]

    #Get complete abstracts
    abstracts = book_complete['plot_summary']

    #Reset index for both abstracts and dataframe
    book_complete = book_complete.reset_index(drop=True)
    abstracts = abstracts.reset_index(drop=True)

    ## Check for non-English books and remove them

    #Nltk word list
    nltk.download('words')
    words = set(nltk.corpus.words.words())

    #Emplty list for percentage of words in English
    perc_eng = []

    #For each abstract, check what percentage of the words
    #are in English
    for i in abstracts:
        k = re.findall(r'\w+', i)
        count = 0
        for j in k:
        if j in words:
            count += 1
        perc_eng.append(count/len(k))
    

    #Remove all plot_summaries where at least
    #40 percent of the words are not in English
    abs_idx = []
    for i in perc_eng:
        if i > 0.4:
            abs_idx.append(True)
        else:
            abs_idx.append(False)
    
    abstracts = abstracts[abs_idx]
    book_complete = book_complete[abs_idx]
    abstracts = abstracts.reset_index(drop=True)
    book_complete = book_complete.reset_index(drop=True)

    return abstracts, book_complete

The function returns the cleaned abstract and completed dataframe which we can now use to build our book recommender. 

# Part 3 - Book Recommender

We will use a pre-trained transformer to produce our recommendations. The transformer is used to create embeddings from the book's plot summaries. Each book is then compared with other books based on the cosine similarity of their plot summary embeddings. Plot summaries that are similar to each other in syntax, subject matter, and semantic meaning will have higher cosine simiarities. The model also incorporates the tags data to see if each book pair shares common tags. These two metrics are used to create a final "similarity" score. 

Before we build our model, let's import the necessary libraries.

    import numpy as np 
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    #!pip install -U sentence-transformers
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
    from sklearn.metrics.pairwise import cosine_similarity

We then initiate our model from Hugging face. For good quality embeddings, HuggingFace suggests using Microsoft's 'paraphrase-MiniLM-L6-v2' model. The model is trained on a variety of text corpuses, including wiki, quora, yahoo answers, stackexchange and so forth. This is a slower model, but since we are using a smaller dataset to start with, it will do nicely. When we incorporate a larger database, we might have to sacrifice some performance for a quicker and lighter model. 

    #Initialize the model
    model = SentenceTransformer('paraphrase-mpnet-base-v2')

Next, we input the plot summaries we extracted from our GoodReads data into the model to get document embeddings. But what is an embedding? One of the key challenges in natural language processing is representing the *meaning* of a text numerically. Previous methods such as one-hot-encoding and term frequency-inverse document frequency (TF-IDF) worked well for some use cases, but had a distinct disadvantage. They could represnt the syntactic structure, but not the semantic information within languages. Embeddings, derived from training models with large text corpuses, solved this problem. Embeddings are fixed vector representations of the text. 

For an illustrated explanation of text embeddings, please visit this excellent [blog post](https://jalammar.github.io/illustrated-word2vec/) by Jay Alammar.

We get the text embeddings for our plot summaries: 

    text_embeddings = model.encode(abstracts, batch_size = len(abstracts), show_progress_bar = True)

For this specific data set, we have 719 books, each of which has a plot summary that is represented by a vector of 768 numbers. So the dimensions of our embeddings is 719 x 768. This will be the data on which our recommendation system is built. Next we sort the documents by their cosine simiarlity. 

    #The cosine simiarity help us determine which abstracts are close 
    #to each other based on the embeddings.
    similarities = cosine_similarity(text_embeddings)

Now we have, for each plot summary, a matrix of how close other plot summaries are in the multi-dimensional space, based on the text embeddings. We simply sort it by their indexes to find which books are closer to other books in terms of the simiarities of their summaries. 

    #Sort the cosine similarity by ascending order
    similarities_sorted = similarities.argsort()

At this point, we will add one more element to the algorithm. We want the user to find that the recommendations are of a similar type to the book they query. The genre tags can help curate our recommendations for this purpose. We will create a helper function that gives us a similar matrix of tags simiarity between books. We'll call this function **tag_ratio**. To help with making the tag_ratio, we also create a second function called **intersection**, which gives us the ratio of common tags by the total tags of the first book. 

    def intersection(lst1, lst2):
        """
            Takes two lists and returns the ratio of number of 
            common elements to the length of the first list.

            :param p1: List 1
            :param p2: List 2
            :return: Return ratio of intersection to length of first list
        """ 
        #Takes two list and returns ratio of intersection length by total length of first list
        try: 
            intsect = len(set(lst1) & set(lst2))/len(lst1)
        except ZeroDivisionError:
            intsect = len(set(lst1) & set(lst2))/1
        return intsect

    def tag_ratio(df):
        """
            Takes in books dataframe and returns the tag ratio for all book pairs.

            :param p1: Dataframe of books
            :return: Returns book-pair wise tag ratio for all books
        """ 
        #Takes in a dataframe and returns a tag ratio list of lists
        all_tags = df.tags 

        tag_ratio_main = []

        for i in all_tags:
            tag_ratio_sub = []
            for j in all_tags:
                tag_ratio_sub.append(intersection(i,j))
            tag_ratio_main.append(tag_ratio_sub)
        
        return(tag_ratio_main)

    tag_ratio_book_final = tag_ratio(book_final)


Now we have everything we need to make our recommendation algorithm. We want receommeder to rely more on the plot summaries than the tag data, and therefore, we will weight the plot summaries data four times as much as the tag data. This is not a hard and fast rule, so we may want to tune the *z-score* to adjust our recommendation algorithm. 

    tag_weight = 0.2
    abstract_weight = 0.8
    z_score = (np.array(similarities)*(abstract_weight) + np.array(tag_ratio_book_final)*(tag_weight))/2

    similarities_sorted_z = z_score.argsort()

Our final simiarity matrix is ready! The last function we have to create is to link the book query to our database and our similarities matrix. 

    def similar_books(title, df, num_books, sorted_list):
    """
    Returns similar books to the book title provided.

    :param p1: Book Title
    :param p2: Dataframe of books to search from
    :param p3: Number of recommendations
    :param p4: The sorted list of books to search from
    :return: Return number of book recommendations equivalent to the num_books
    """

    idx_num = df[df["title"] == title].index.values
    return df["title"][sorted_list[idx_num][0][-2:-(num_books+2):-1]].values

Now when we search for a book title, the **similar books** will return the *num_books* number of books closest in simiarity from our dataset. This concludes our book recommender section! In the next section, we will see how to deploy this model to a web app. 

# Part 4 - Deploy to Heroku

Using our deployed web app, the recommendations look like the following: 

![image](https://user-images.githubusercontent.com/59543579/122615577-beb17d80-d056-11eb-8a50-36432793edba.png)

To check out the data science part of this project, please see this [repo.](https://github.com/rasim321/Book_Recommender)

To see the web development part, visit this [one.](https://github.com/rasim321/GR_bookrec_heroku)


