#!/bin/sh

# entrypoint.sh

# Set the necessary environment variables
export FLASK_APP=vegi_esc_api.app:gunicorn_app  
export FLASK_ENV=production  # Set the desired Flask environment (e.g., "development", "production")

echo "Entrypoint for app run with HOME=\"$HOME\""
echo "Copying \"/.chromadb\" to \"$HOME/.chromadb\""
cp /.chromadb $HOME/.chromadb -r
echo "Copying \"/env\" to \"$HOME/env\""
cp /env $HOME/env -r

# ~ https://stackoverflow.com/a/46933153
export PATH="$PATH:/env/bin:$HOME/env/bin"
# source activate $HOME/env
export PATH="$HOME/python/bin:$PATH"


# Start Gunicorn
# NOTE: We have set workers to 1 to fix concurrency issues for now with LLM's connection to chromadb instance
# exec /env/bin/gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 vegi_esc_api.app:gunicorn_app
# ! heroku web local will always fail from within mac as the
# * use heroku run web to launch the web dyno ~ https://devcenter.heroku.com/articles/dynos#cli-commands-for-dyno-management
exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 600 vegi_esc_api.app:gunicorn_app

