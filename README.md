app
============

Simple web service providing a word embedding API. The methods are based on Gensim Word2Vec implementation. Models are passed as parameters and must be in the Word2Vec text or binary format. Updated to run on Python 3.
# Install Dependencies 
## Using conda 
```
conda env create --file environment.yaml
```
## Using pip 
* See Notion Docs -> [venvs in pip](https://www.notion.so/gember/Python-Cheats-8d7b0cc6f58544ef888ea36bb5879141?pvs=4#03585a911a79487db4004f0f8640b9c6)
```
pip install -r requirements.txt
```

# Launching the service
```
python app --model path/to/the/model [--host host --port 1234]
```
or   
```
python app.py --model /path/to/GoogleNews-vectors-negative300.bin --binary BINARY --path /word2vec --host 0.0.0.0 --port 5000
```



## Example calls
```
curl http://127.0.0.1:5000/n_similarity?ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant
curl http://127.0.0.1:5000/similarity?w1=Sushi&w2=Japanese
curl http://127.0.0.1:5000/most_similar?positive=indian&positive=food[&negative=][&topn=]
curl http://127.0.0.1:5000/model?word=restaurant
curl http://127.0.0.1:5000/model_word_set
```

Note: The "model" method returns a base64 encoding of the vector. "model\_word\_set" returns a base64 encoded pickle of the model's vocabulary. 

## Where to get a pretrained model

See pretrained models [here](https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models)

In case you do not have domain specific data to train, it can be convenient to use a pretrained model. 
Please feel free to submit additions to this list through a pull request.
 
 
| Model file | Number of dimensions | Corpus (size)| Vocabulary size | Author | Architecture | Training Algorithm | Context window - size | Web page |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/) | 300 |Google News (100B) | 3M | Google | word2vec | negative sampling | BoW - ~5| [link](http://code.google.com/p/word2vec/) |
| [Freebase IDs](https://docs.google.com/file/d/0B7XkCwpI5KDYaDBDQm1tZGNDRHc/edit?usp=sharing) | 1000 | Gooogle News (100B) | 1.4M | Google | word2vec, skip-gram | ? | BoW - ~10 | [link](http://code.google.com/p/word2vec/) |
| [Freebase names](https://docs.google.com/file/d/0B7XkCwpI5KDYeFdmcVltWkhtbmM/edit?usp=sharing) | 1000 | Gooogle News (100B) | 1.4M | Google | word2vec, skip-gram | ? | BoW - ~10 | [link](http://code.google.com/p/word2vec/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 50 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 100 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 200 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia+Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip) | 300 | Wikipedia+Gigaword 5 (6B) | 400,000 | GloVe | GloVe | AdaGrad | 10+10 | [link](http://nlp.stanford.edu/projects/glove/) |
| [Common Crawl 42B](http://nlp.stanford.edu/data/glove.42B.300d.zip) | 300 | Common Crawl (42B) | 1.9M | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Common Crawl 840B](http://nlp.stanford.edu/data/glove.840B.300d.zip) | 300 | Common Crawl (840B) | 2.2M | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 25 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 50 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 100 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Twitter (2B Tweets)](http://www-nlp.stanford.edu/data/glove.twitter.27B.zip) | 200 | Twitter (27B) | ? | GloVe | GloVe | GloVe | AdaGrad | [link](http://nlp.stanford.edu/projects/glove/) |
| [Wikipedia dependency](http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2) | 300 | Wikipedia (?) | 174,015 | Levy \& Goldberg | word2vec modified | word2vec | syntactic dependencies | [link](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) |
| [DBPedia vectors (wiki2vec)](https://github.com/idio/wiki2vec/raw/master/torrents/enwiki-gensim-word2vec-1000-nostem-10cbow.torrent) | 1000 | Wikipedia (?) | ? | Idio | word2vec | word2vec, skip-gram | BoW, 10 | [link](https://github.com/idio/wiki2vec#prebuilt-models) |
| [60 Wikipedia embeddings with 4 kinds of context](http://vsmlib.readthedocs.io/en/latest/tutorial/getting_vectors.html#) | 25,50,100,250,500 | Wikipedia | varies | Li, Liu et al. | Skip-Gram, CBOW, GloVe | original and modified | 2 | [link](http://vsmlib.readthedocs.io/en/latest/tutorial/getting_vectors.html#) |
| [German Wikipedia+News](http://cloud.devmount.de/d2bc5672c523b086/german.model) | 300 | Wikipedia + Statmt News 2013 (1.1B) | 608.130 | Andreas Müller | word2vec | Skip-Gram | 5 | [link](https://devmount.github.io/GermanWordEmbeddings/)


# Deploying to Heroku

## Procfile

We create a Procfile to tell heroku how to host the flask application.

We use gunicorn to manage creating the instance of flask

We have the file containing the root function for our flask app in `app.py`

This is reflected with a Procfile as so:
```
web: gunicorn app:app
```

See a how to [here](https://evancalz.medium.com/deploying-your-flask-app-to-heroku-43660a761f1c)

## Helpful Scripts

See scripts in ./shell_scripts_devops/*.sh



``` shell
conda activate vegi-esc-api
conda env export --from-history > environment.yml

conda-lock -f predict-environment.yml -f pot-environment.yml -p linux-64 -k explicit --filename-template "predict-{platform}.lock"

zsh ./shell_scripts_devops/makeRun.sh # ~ see https://docs.docker.com/engine/reference/commandline/buildx_build/
# or manually run:
# make distroless-buildkit
# imageName="vegi_esc_server_distroless-buildkit"
# docker run --platform linux/amd64 -it -p 2001:5002 $imageName
# open localhost:2001/success/fenton

! if see: no space left on device error, then run: `docker system prune`


```


### Dash Endpoints:

- http://localhost:5002/dashboard/ or http://127.0.0.1:5002/dashboard/ and http://127.0.0.1:5002/ to login with username and pass defined in .env. The dashboard is where the code is defined in layouts for dash apps.
### All other Endpoints defined in `app.py` with port `2001` for when in docker image, and `5002` when running without docker
### Localhost Port
- http://localhost:5002/success/fenton
- http://127.0.0.1:5002/n_similarity?ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant -> { Success: 0.7014271020889282 }
- http://127.0.0.1:5002/similarity?w1=Sushi&w2=Japanese -> { Success: 0.3347574472427368 }
- http://localhost:2001/similarity?w1=bike&w2=car -> { Success: 0.7764649391174316 } 
- http://127.0.0.1:5002/most_similar?positive=indian&positive=food[&negative=][&topn=] -> Internal Server Error!
- http://127.0.0.1:5002/model?word=restaurant -> { Success: AAAYvgAA/r0AACk9AABWPgAA+b0AAFc+AABjvQAAVb0AAGs+AAClPQAA8D0AAFQ8AAA9PgAAs74AADu9AAC1PQAAqb0AAFs8AACXPQAAmL0AAEo+AAAxvgAAqjwAABm+AABvvAAA8r0AAJc9AAAuPgAAtT4AAO89AAAqvgAAer4AAKE9AAA+PgAAM74AALC9AAD6PgAAHz0AAAy+AADhvQAALr4AAGq9AACwPQAApj4AAL+9AABsvgAA1b4AAGm7AABNPQAAcj4AAKM8AAA+PgAAnj0AAFy+AADcvQAAlrwAAF0+AACxPQAA9L0AACu+AAAFvgAAED0AAG6+AABevgAADD4AAGs9AAAvPQAAUr4AAIU+AABEvgAAsD0AAAI+AAAMPQAAiL0AAOK+AADgvQAA474AAFo+AAA3OwAAh74AAAo+AACZvgAAQr4AANM9AACuvgAA1jwAAKy+AACrPgAAlD0AAK6+AAAcvQAAn74AAIe+AAAJvQAAOb4AAAS+AAAOvgAACb4AABc+AADuvQAAn74AADS+AABdPgAAJDwAAHY+AABgvgAAwD0AACy+AAC4vAAAAL4AAGO+AAAjPQAAjb0AAJ6+AAAwPgAAX74AALM9AADXvQAAPr4AAJC9AABLvgAAyb0AABe9AACaPQAAVj0AADw9AACuvAAAeDwAAAM9AAAzPQAAvD0AAKW9AACCPgAAET4AAIu9AACMPQAApT0AAI29AADfPQAABT4AAD2+AAD+vQAAPL4AAEK9AAC0PAAApb0AAFu+AAArPgAAWL4AAGY+AAAlPgAAuD4AAHA9AAC3PQAA7jwAAJk7AAAZPQAAeD0AAAW+AACuvQAAab4AAEc+AADxuwAAiL0AAJu9AAAEvgAAOr4AADC9AADWPQAAaL4AAGa+AACtvQAAiD0AAHG+AABJvgAASj4AAIo+AACQPgAAgT4AAAk+AADSvQAAyz0AAB49AACQvgAA+D0AAPq9AAAOvQAA1j4AAKe9AACwvgAADj0AAEM+AABuvQAAPT4AAMy9AAAwPgAAkD0AAKC7AAA8PAAANb0AAIW+AABFPgAAtDoAADs7AADkvAAAW74AANs9AAA+PgAAkzwAAN47AACRvQAAwz0AAC49AAAiPgAA570AAHE+AAD3vQAAvjwAACG+AAA0PQAAFT4AAKU9AACUvQAAwLwAADo+AAAgPgAAM74AAHo+AABVvgAArLwAACM+AAA4vgAA0b4AAHY+AABTPgAAZr4AAJo+AADiPQAAPb4AAKs9AAB4PgAA8b0AADA+AACmvAAAKr0AAI2+AACxPgAApL0AADc+AAC+PQAAGb0AACM+AAB0PgAAQr4AANW9AACQvgAAgz0AAD29AAB5vgAARD0AAJa9AADlOwAArTwAAMY9AACcPQAAKz0AAHu+AABnvgAAsb0AAAA8AAB3vQAAAb0AAJe+AAAePQAADz0AAGc8AAD7vQAAEr4AAEs+AADyvgAAq7wAAAk+AAAfvgAAYD4AAEW+AAAZvQAAO74AAHI+AAAivgAA7D4AAOQ8AABuPQAAVDwAAPc+AADEPgAAyr0AAHY9AADCvQAADT4AAOo9 }
- http://127.0.0.1:5002/model_word_set -> Internal Server Error!
- http://127.0.0.1:2001/nearest-word-in-model?w1=bike

### Docker Port
- http://127.0.0.1:2001/success/fenton
- http://127.0.0.1:2001/vegi-users -> Success
- http://127.0.0.1:2001/sentence_similarity?s1=Chocolate%20cake&s2=Bakery -> Success: 0.7233545909583172 using full googlenews model
- http://127.0.0.1:2001/rate-vegi-product/2 -> Success
- http://127.0.0.1:2001/rate-latest?n=1 -> 
