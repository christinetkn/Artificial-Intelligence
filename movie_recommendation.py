#####################################################################################
################### Mavraeidi Lydia 03117181 ########################################
################### Tsakanika Christina 03117012 ####################################
#####################################################################################
# this file is in the same directory as movies_metadata.csv, test_ratings.csv, train_ratings.csv
# we ran this file in Ubuntu 18.04 LTS by the command python3 movies.py

import pandas as pd
from pyswip import Prolog
import numpy as np

def clean_text(text):
  text = text.replace(u'\xa0', u'')
  text = text.replace(u"'", u'')
  text = text.replace(u'"', u'')
  return text

import pandas as pd 
data = pd.read_csv("movies_metadata.csv") 

data.fillna("UNK", inplace=True)
data.head()


prolog = Prolog()


movies = []
literals = []

movie_score = {}
for row in data.itertuples(index=True, name='Pandas'):
  movie_title = clean_text(getattr(row, 'movie_title'))
  movies.append(movie_title)
  
  actor1 = getattr(row, 'actor_1_name')
  c =  clean_text(actor1)
  literals.append("actor('"+ movie_title +"','"+ c +"')")
  
  actor2 = getattr(row, 'actor_2_name')
  literals.append("actor('"+ movie_title +"','"+ clean_text(actor2) +"')")
  
  actor3 = getattr(row, 'actor_3_name')
  literals.append("actor('"+ movie_title +"','"+ clean_text(actor3) +"')")

  language = getattr(row, 'language')
  literals.append("language('"+ movie_title +"','"+ language +"')")
  
  director = getattr(row, 'director_name')
  literals.append("director('"+ movie_title +"','"+ clean_text(director) +"')")
  
  for keyword in getattr(row, 'plot_keywords').split("|"):
    literals.append("keyword('"+ movie_title +"','"+ clean_text(keyword) +"')")
    if (keyword == "black and white" or keyword == "black and white scene" 
        or keyword == "black and white to color"):
      literals.append("bnw('"+ movie_title +"',"+"'black and white'"+")") 
  
  for genre in getattr(row, 'genres').split("|"):
    literals.append("genre('"+ movie_title +"','"+ genre +"')")
  
literals.sort()
for literal in literals:
  prolog.assertz(literal)

#all the movies that have one common genre:
prolog.assertz('''(genre1(X,G1):- genre(X, G1))''')

#all the movies that have two common genres:
prolog.assertz('''(genre2(X, G1, G2) :- genre(X,G1), genre(X, G2), G1\=G2)''')

#all the movies that have three common genres:
prolog.assertz('''(genre3(X, G1, G2, G3):- genre(X, G1), genre(X, G2), genre(X, G3),
                   G1\=G2, G1\=G3, G2 \= G3)''')

#all the movies that have four common genres:
prolog.assertz('''(genre4(X, G1, G2, G3, G4):- genre(X, G1), genre(X, G2), genre(X, G3), genre(X, G4),
                   G1\=G2, G1\=G3, G1\=G4, G2\=G3, G2\=G4, G3\=G4)''')
#all the movies that have five common genres:
prolog.assertz('''(genre5(X, G1, G2, G3, G4, G5):- genre(X, G1), genre(X, G2), genre(X, G3), genre(X, G4), genre(X, G5),
                   G1\=G2, G1\=G3, G1\=G4, G1\=G5, G2\=G3, G2\=G4, G2\=G5, G3\=G4, G3\=G5, G4\=G5)''')
#same director
prolog.assertz('''(same_director(Movie,D):- director(Movie, D))''')

#same language
prolog.assertz('''(same_language(Movie,L):- language(Movie, L))''')

#all the movies that have one common actor:
prolog.assertz('''(actor1(X,A1):- actor(X, A1))''')

#all the movies that have two common actors:
prolog.assertz('''(actor2(X, A1, A2) :- actor(X,A1), actor(X, A2), A1\=A2)''')

#all the movies that have three common actors:
prolog.assertz('''(actor3(X, A1, A2, A3):- actor(X, A1), actor(X, A2), actor(X, A3), A1\=A2, A1\=A3, A2 \= A3)''')

#all the movies that have one common plot keyword:
prolog.assertz('''(keyword1(X,K1):- keyword(X, K1))''') 

#all the movies that have two common plot keywords:
prolog.assertz('''(keyword2(X, K1, K2) :- keyword(X,K1), keyword(X, K2), K1\=K2)''')

#all the movies that have three common plot keywords:
prolog.assertz('''(keyword3(X, K1, K2, K3):- keyword(X, K1), keyword(X, K2), keyword(X, K3),
                   K1\=K2, K1\=K3, K2 \= K3)''')

#all the movies that have four common plot keywords:
prolog.assertz('''(keyword4(X, K1, K2, K3, K4):- keyword(X, K1), keyword(X, K2), keyword(X, K3), keyword(X, K4),
                   K1\=K2, K1\=K3, K1\=K4, K2\=K3, K2\=K4, K3\=K4)''')
#all the movies that have five common plot keywords:
prolog.assertz('''(keyword5(X, K1, K2, K3, K4, K5):- keyword(X, K1), keyword(X, K2), keyword(X, K3), keyword(X, K4), keyword(X, K5),
                   K1\=K2, K1\=K3, K1\=K4, K1\=K5, K2\=K3, K2\=K4, K2\=K5, K3\=K4, K3\=K5, K4\=K5)''')

#Here we print some examples

i =0

Q = ["genre3(X, 'Crime', 'Action', 'Adventure')", "genre5(X, 'Action', 'Drama', 'Horror', 'Science Fiction', 'Thriller')", "same_director(X, 'Christopher Nolan')", "same_language(X, 'Italiano')"
, "actor1(X, 'George Clooney')","bnw(X, 'black and white')", "keyword2(X, 'superhero', 'revenge')", "keyword3(X, 'spy', 'secret agent', 'british secret service')"]

P = ["All the action adventure crime movies are: ", "All the Action, Drama, Horror, Science Fiction, Thriller movies are:",
     "Christopher Nolan has directed:","All the Italian movies are: ",
     "George Clooney starred movies:","All the black and white movies are:","Movies with superhero and revenge keywords:", "Movies with keywords: spy, secret agent, british secret service: "]

for i in range(len(Q)):
  q = prolog.query(Q[i])
  s = set()
  for soln in q:
    m = soln["X"] 
    if m not in s:
      s.add(soln["X"])
  print(P[i])
  print(s,'\n') 

###################################
#2 Recommendation System
###################################

prolog.assertz('''(find_similar_movies_1(X, Y):- genre1(X,G1), genre1(Y,G1), X \= Y)''')

prolog.assertz('''(find_similar_movies_2(X, Y):- genre2(X,G1,G2), genre2(Y,G1,G2), X \= Y; actor1(X,A1), actor1(Y, A1), X \= Y)''')

prolog.assertz('''(find_similar_movies_3(X, Y):- genre3(X,G1,G2,G3), genre3(Y,G1,G2,G3), X \= Y; actor2(X,A1,A2), actor2(Y, A1, A2), X \= Y )''')

prolog.assertz('''(find_similar_movies_4(X, Y):- genre4(X,G1,G2,G3,G4), genre4(Y,G1,G2,G3,G4), X \= Y; actor2(X,A1,A2), actor2(Y, A1, A2), keyword1(X,K1), keyword1(Y,K1), X \= Y )''')

prolog.assertz('''(find_similar_movies_5(X, Y):- genre5(X,G1,G2,G3,G4,G5), genre5(Y,G1,G2,G3,G4,G5), X \= Y; actor2(X,A1,A2), actor2(Y, A1, A2), keyword2(X,K1,K2), keyword2(Y,K1,K2), X \= Y)''')

def simple_recommender(movie):
    s = []
    
    q = prolog.query("find_similar_movies_5('" + movie +"',M)")
    for soln in q:
      m = soln['M'] 
      if m not in s:
        s.append(soln['M'])
    
    q = prolog.query("find_similar_movies_4('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
          s.append(soln['M'])
    
    q = prolog.query("find_similar_movies_3('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])
    
    q = prolog.query("find_similar_movies_2('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])
    
    q = prolog.query("find_similar_movies_1('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])

    q.close()
    answers = s
    return answers

c = simple_recommender('Avatar')[:50]
print('Simple recommender for movie Avatar', c)

################################################################
#3 Recommendation System User Preferences
################################################################
def new_simple_recommender(movie):
    s = []
    level = []
    q = prolog.query("find_similar_movies_5('" + movie +"',M)")
    for soln in q:
      m = soln['M'] 
      if m not in s:
        s.append(soln['M'])
        level.append(4) #addition: index of score_weights
    
    q = prolog.query("find_similar_movies_4('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
          s.append(soln['M'])
          level.append(3)  #addition: index of score_weights
    
    q = prolog.query("find_similar_movies_3('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])
            level.append(2)  #addition: index of score_weights
    
    q = prolog.query("find_similar_movies_2('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])
            level.append(1)  #addition: index of score_weights
    
    q = prolog.query("find_similar_movies_1('" + movie +"',M)")
    for soln in q:
        m = soln['M'] 
        if m not in s:
            s.append(soln['M'])
            level.append(0)  #addition: index of score_weights

    q.close()
    answers = s
    return answers, level

from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


rating_weights = {0: -1, 1: -0.5, 2:0, 3:0, 4:0.5, 5:1}
score_weights = {0: 1, 1: 2, 2: 4, 3: 6, 4: 8}  # ανάλογα με τα επίπεδα ομοιότητας που έχουν οριστεί στην simple_recommender

def train_recommender(ratings, rating_weights, score_weights, start=0, end = -1, movie_score = {}):
    
    if end == -1:
        end = len(ratings)

    ratings = ratings.iloc[range(start, end)]
    for row in tqdm(ratings.itertuples(index=True, name='Pandas')):
        movie = clean_text(getattr(row, 'movie_title'))
        rating = getattr(row, 'rating')

        similar_movies, score_index = new_simple_recommender(movie) #addition

        for i in range(len(similar_movies)):
            if similar_movies[i] not in movie_score:
                movie_score[similar_movies[i]] = rating_weights[int(rating)] * score_weights[score_index[i]]
            else:
                movie_score[similar_movies[i]] += rating_weights[int(rating)] * score_weights[score_index[i]] #το weight θα το ορίσετε ανα επίπεδο ομοιότητας οι πολύ όμοιες ταινίες θα έχουν μεγαλύτερο βάρος
    return movie_score



def predict_example(ratings, movie_score):
    real, pred = [], []
    for i, row in enumerate(ratings.itertuples(index=True, name='Pandas')):
        movie = clean_text(getattr(row, 'movie_title'))
        rating = getattr(row, 'rating')

        if movie in movie_score: 
            pred.append(int(movie_score[movie] > 0)) 
            real.append(int(rating > 3))
        else: 
            pred.append(0)
            real.append(int(rating > 3))

    return real, pred


def get_metrics(real, pred):
    metrics = {}
    metrics["precision"] = precision_score(real, pred)
    metrics["recall"] = recall_score(real, pred)
    metrics["f1"] = f1_score(real, pred)
    return metrics

train_ratings = pd.read_csv("train_ratings.csv")
test_ratings = pd.read_csv("test_ratings.csv")

movie_score = train_recommender(train_ratings, rating_weights, score_weights, 0, 10)
real, pred = predict_example(test_ratings, movie_score)
print(get_metrics(real, pred))

movie_score = train_recommender(train_ratings, rating_weights, score_weights, 10, 20, movie_score)
real, pred = predict_example(test_ratings, movie_score)
print (get_metrics(real, pred))

movie_score = train_recommender(train_ratings, rating_weights, score_weights, 20, 30, movie_score)
real, pred = predict_example(test_ratings, movie_score)
print (get_metrics(real, pred))

movie_score = train_recommender(train_ratings, rating_weights, score_weights, 30, 50, movie_score)
real, pred = predict_example(test_ratings, movie_score)
print(get_metrics(real, pred))

movie_score = train_recommender(train_ratings, rating_weights, score_weights,start = 50, movie_score= movie_score)
real, pred = predict_example(test_ratings, movie_score)
print (get_metrics(real, pred))
print('Very Merry Christmas and Happy Holidays!')



