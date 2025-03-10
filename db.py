# library
import os
import ssl

import sqlalchemy


#  connect unix socket - cloud proxy
def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    """Initializes a Unix socket connection pool
    for a Cloud SQL instance of Postgres."""
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.
    db_user = os.environ["DB_USER"]  # e.g. 'my-database-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-database-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
    unix_socket_path = os.environ[
        "INSTANCE_UNIX_SOCKET"
    ]  # e.g. '/cloudsql/project:region:instance'

    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # postgresql+pg8000://<db_user>:<db_pass>@/<db_name>
        #                         ?unix_sock=<INSTANCE_UNIX_SOCKET>/.s.PGSQL.5432
        # Note: Some drivers require the `unix_sock` query parameter to
        # use a different key.
        # For example, 'psycopg2' uses the path set
        # to `host` in order to connect successfully.
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=db_user,
            password=db_pass,
            database=db_name,
            query={"unix_sock": f"{unix_socket_path}/.s.PGSQL.5432"},
        ),
        # ...
    )
    return pool


#  connect tcp socket cloud proxy
def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    """Initializes a TCP connection pool for a Cloud SQL instance of Postgres."""
    db_host = os.environ[
        "INSTANCE_HOST"
    ]  # e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
    db_port = os.environ["DB_PORT"]  # e.g. 5432

    connect_args = {}
    if os.environ.get("DB_ROOT_CERT"):
        connect_args = {}
        db_root_cert = os.environ["DB_ROOT_CERT"]
        db_cert = os.environ["DB_CERT"]  # e.g. '/path/to/my/client-cert.pem'
        db_key = os.environ["DB_KEY"]  # e.g. '/path/to/my/client-key.pem'

        ssl_context = ssl.SSLContext()
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.load_verify_locations(db_root_cert)
        ssl_context.load_cert_chain(db_cert, db_key)
        connect_args["ssl_context"] = ssl_context

    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,  # type: ignore
            database=db_name,
        ),
        connect_args=connect_args,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,  # 30 seconds
        pool_recycle=1800,  # 30 minutes
    )
    return pool
