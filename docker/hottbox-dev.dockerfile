FROM hottbox-dev-base

COPY . ${HOME}
RUN pip install --no-cache-dir -e '.[tests, docs]'