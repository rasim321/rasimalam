---
layout: post
title:  "Book Recommender"
date:   2021-06-10 16:16:52 -0400
category: project
img: ../img/bookrec.jpg
excerpt: "Book recommender that uses plot summaries from GoodReads and embeddings from Google's BERT model to recommend books."
github: "https://github.com/rasim321/Book_Recommender"
---

<img src="\rasimalam\img\bookrec.jpg" />

Check out the Book Recommender [here](https://gr-bookrec.herokuapp.com) 

The Book Recommender uses metadata scraped from [GoodReads](https://www.goodreads.com) to recommend interesting and relevant books. 

The web scraper collects title, author, reviews, ratings, plot summaries, and tags data to create the website's database. Google's BERT transformer is then used to create embeddings from the book's plot summaries. Each book is then compared with other books based on the cosine similarity of their plot summary embeddings. Plot summaries that are similar to each other in syntax, subject matter, and semantic meaning will have higher cosine simiarities. The model also incorporates the tags data to see if each book pair shares common tags. These two metrics are used to create a final "similarity" score. 

Finally, the similarity scores are then used to recommend books that match both the genre and plot similarity to the book queried. 



The recommender currently has around 700 books at the pilot stage. When users input new books that are not in the database, they are automatically downloaded and added to the data. This script runs once per day to gradually improve the recommender and add new sought-after titles. 

To check out the data science part of this project, please see this [repo.](https://github.com/rasim321/Book_Recommender)

To see the web development part, visit this [one.](https://github.com/rasim321/GR_bookrec_heroku)


