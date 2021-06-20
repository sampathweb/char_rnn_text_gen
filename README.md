# char_rnn_text_gen

Text Generation RNN

## Commands

```bash
# Create Virtual Env
virtualenv env
source env/bin/activate
pip install -r requirements.txt
# Run locally to test
gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app

# Deploy to Google Cloud
gcloud app deploy
```
