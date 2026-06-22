# Base image maintained by Apify (see https://hub.docker.com/r/apify/).
FROM apify/actor-python:3.14

USER myuser

# Copy requirements first so the dependency layer is cached across source changes.
COPY --chown=myuser:myuser requirements.txt ./

RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze

# Copy the rest of the source code.
COPY --chown=myuser:myuser . ./

# Ensure the Actor code compiles.
RUN python -m compileall -q news_portal/

# Launch the Actor.
CMD ["python", "-m", "news_portal"]
