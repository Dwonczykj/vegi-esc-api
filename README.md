app
============

Simple web service providing a word embedding API. The methods are based on Gensim Word2Vec implementation. Models are passed as parameters and must be in the Word2Vec text or binary format. Updated to run on Python 3.

# Resources
- [Joey's Notion Article](https://www.notion.so/gember/Deploying-conda-environments-in-Docker-containers-how-to-do-it-right-Uwe-s-Blog-6709ea7cd2b14756bb7597a4ed70bb34#d72953bfddab4e5c86ac4de53bf4abbd)

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

See scripts in `./shell_scripts_devops/*.sh`

To rebuild the docker image and push it to heroku, run `push_heroku.sh`

Alternatively, the old method that builds the container and then runs it locally is below:

NOTE: We use the predict-environment.yml file to remove packages from the full environment.yml that are only needed for development and not production. [Linked](https://www.notion.so/gember/Deploying-conda-environments-in-Docker-containers-how-to-do-it-right-Uwe-s-Blog-6709ea7cd2b14756bb7597a4ed70bb34?pvs=4#8d6afb50984347f0911c899a563ccb07)

NOTE on conda-lock: Then, with predict-environment.yml created, instead of using that as an input for the container build, we are using conda-lock on the predict-environment.yml file to render the requirements into locked pinnings for different architectures. This enables us to have a consistent environment to make developer as well as production environments reproducible. Another benefit of using conda-lock is that you can generate the lockfile for any platform from any other platform. This means we can develop on macOS and have production systems on Linux but still generate the lockfiles for all of them on either systems.

``` shell
conda activate esc-llm  # * i have cloned a backup named esc-llm-backup-2023-07-08
conda env export --from-history > environment.yml
cp environment.yml predict-environment.yml

# conda-lock -f predict-environment.yml -f pot-environment.yml -p osx-64 -k explicit --filename-template "predict-{platform}.lock"
conda-lock -f predict-environment.yml -f pot-environment.yml -p linux-64 -k explicit --filename-template "predict-{platform}.lock"

docker system prune -f

zsh ./shell_scripts_devops/makeRun.sh # ~ see https://docs.docker.com/engine/reference/commandline/buildx_build/
# or manually run:
# make distroless-buildkit
# imageName="vegi_esc_server_distroless-buildkit"
# docker run --platform linux/amd64 -it -p 2001:5002 $imageName
# open localhost:2001/success/fenton

# ! For errors: "no space left on device" error, then run: `docker system prune`


```

*The docker build command takes about 13 mins to run on localhost.*

It is run within the makeRun.sh script called above

*NOTE: Our `setup.py` file that is called to install a module into `/pkg` in the `dockerfile` has the package named to `vegi-esc-api` using the `name="vegi_esc_api"` field in `setup()` in the `setup.py` file. Note how '_' are replaced with '-'. So this `vegi-esc-api` will precede module names for relative imports and calling the app module from gunicorn command*

# Running the container locally

### Run a shell to inspect the target container:
```sh
cd ./vegi-esc-api
dockerFileExt=distroless-buildkit
imageName=vegi_esc_server_$dockerFileExt
condaVenvName=esc-llm
dockerFileName=docker/Dockerfile.$dockerFileExt
herokuWebAppName=vegi-esc-server
docker run \                    
    --env DATABASE_HOST=host.docker.internal \
    --env-file ./.env \
    --platform linux/amd64 \
    -p 2001:5001 \
    -it --entrypoint=sh \
    $imageName
```
### Run the default entrypoint on target container:
```sh
cd ./vegi-esc-api
dockerFileExt=distroless-buildkit
imageName=vegi-esc-server-$dockerFileExt
condaVenvName=esc-llm
dockerFileName=docker/Dockerfile.$dockerFileExt
herokuWebAppName=vegi-esc-server
docker run \                    
    --env DATABASE_HOST=host.docker.internal \
    --env-file ./.env \
    --platform linux/amd64 \
    -p 2001:5001 \
    -it \
    $imageName
```


### Dash Endpoints:

- http://localhost:5002/dashboard/ or http://127.0.0.1:5002/dashboard/ and http://127.0.0.1:5002/ to login with username and pass defined in .env. The dashboard is where the code is defined in layouts for dash apps.
- http://127.0.0.1:5001/login/?next=%2Fdashboard%2F

*All other Endpoints defined in `app.py` with port `2001` for when in docker image, and `5002` or `5001` when running without docker*
## Localhost EndPoints
### Connection status
- http://localhost:5002/success/fenton
### LLM Admin
- http://127.0.0.1:5002/reset-llm-db?reinit=True
### LLM Queries
- http://127.0.0.1:5002/llm/view-vector-store
- http://127.0.0.1:5002/llm/view-vector-store-documents
- http://127.0.0.1:5002/llm/query-vector-store?query=Hummous
### ESC Endpoints
- http://127.0.0.1:5002/vegi-users -> Success
- http://127.0.0.1:5002/rate-vegi-product/2 -> Success
- http://127.0.0.1:5002/rate-latest?n=1 -> 
### ESC DB Admin Endpoints
- POST http://127.0.0.1:5002/products/add
- POST http://127.0.0.1:5002/explanations/add
### ESC Sources Endpoints
- http://127.0.0.1:5002/sustained/refresh
- http://127.0.0.1:5002/

### Deprecated word2vec specific end points...
- http://127.0.0.1:5002/n_similarity?ws1=Sushi&ws1=Shop&ws2=Japanese&ws2=Restaurant -> { Success: 0.7014271020889282 }
- http://127.0.0.1:5002/similarity?w1=Sushi&w2=Japanese -> { Success: 0.3347574472427368 }
- http://localhost:5002/similarity?w1=bike&w2=car -> { Success: 0.7764649391174316 } 
- http://127.0.0.1:5002/most_similar?positive=indian&positive=food[&negative=][&topn=] -> Internal Server Error!

## Docker End Points
- http://127.0.0.1:2001/success/fenton
- http://127.0.0.1:2001/llm/view-vector-store -> 
- http://127.0.0.1:2001/vegi-users -> Success
- http://127.0.0.1:2001/rate-vegi-product/2 -> Success
- http://127.0.0.1:2001/rate-latest?n=1 -> 
- http://127.0.0.1:2001/llm/view-vector-store-documents -> 
- http://127.0.0.1:2001/llm/query-vector-store?query=hummous -> 
- http://127.0.0.1:2001/sentence_similarity?s1=Chocolate%20cake&s2=Bakery -> Success: 0.7233545909583172 using full googlenews model

# Docker Image IP & Container Port
First get the name of the docker image using Docker Desktop i.e. tender_fermat below
```shell
docker inspect \
  -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' tender_fermat
```
which outputs: `172.17.0.2`
then call: http://172.17.0.2:5002/success/fenton

or
First get the container ID:
```shell
docker ps
```
```shell
docker inspect <container ID>
```
```shell
docker inspect <container id> | grep "IPAddress"
```

# Code Documentation (vegi-esc-api)

Start by reviewing the api endpoints in `vegi-esc-api/src/vegi_esc_api/app.py`.


<!-- Originally referenced from [here](https://github.com/3Top/word2vec-api.git) -->


