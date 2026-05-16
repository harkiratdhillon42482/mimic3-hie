FROM yottadb/yottadb:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev gcc git libffi-dev && rm -rf /var/lib/apt/lists/*

RUN export ydb_dist=/opt/yottadb/current && export ydb_gbldir=/data/yottadb.gld && export ydb_routines="/opt/yottadb/current/libyottadbutil.so" && git clone --depth 1 https://gitlab.com/YottaDB/Lang/YDBPython.git /tmp/YDBPython && cd /tmp/YDBPython && pip3 install --break-system-packages --no-cache-dir . && rm -rf /tmp/YDBPython

RUN pip3 install --break-system-packages --no-cache-dir psycopg2-binary pandas tqdm python-dotenv

RUN echo ". /opt/yottadb/current/ydb_env_set" >> /root/.bashrc

ENV ydb_dist=/opt/yottadb/current
WORKDIR /project
CMD ["/bin/bash"]
