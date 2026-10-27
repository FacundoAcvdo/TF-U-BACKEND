from fastapi import FastAPI

from clustering.main import app as clustering_app
from topicmodeling.main import app as topicmodeling_app
from wordcloud.main import app as wordcloud_app
from eda.main import app as eda_app
from emotions.main import app as emotions_app
from risk.main import app as risk_app

app = FastAPI(title="Analysis Service")

# Keep the individual public route contracts while serving all analysis workloads
# from one process and one container.
app.mount("/clustering", clustering_app)
app.mount("/topicmodeling", topicmodeling_app)
app.mount("/wordcloud", wordcloud_app)
app.mount("/eda", eda_app)
app.mount("/emotions", emotions_app)
app.mount("/suicide", risk_app)
