# from paramiko.util import ClosingContextManager
# from paramiko.transport import Transport
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url
from flask import Flask
from paramiko.ssh_exception import (
    SSHException,
    BadHostKeyException,
    NoValidConnectionsError,
)
from errno import ECONNREFUSED, EHOSTUNREACH
from paramiko.agent import Agent
from paramiko.rsakey import RSAKey
from paramiko.hostkeys import HostKeys
from paramiko.ed25519key import Ed25519Key
from paramiko.ecdsakey import ECDSAKey
from paramiko.dsskey import DSSKey
from paramiko.config import SSH_PORT
from paramiko.common import DEBUG
import paramiko
from pprint import pprint
import socket
import time
from datetime import datetime, timedelta
import boto3
import io
import sqlite3
import pymysql
import os
from typing import Callable, TypeVar, Generic, Any
from functools import wraps, cache
from vegi_esc_api.logger import info, LOG_LEVEL, warn, error
import vegi_esc_api.logger as Logger
from vegi_esc_api.select_vendor_products_with_esc import select_products_sql


T = TypeVar('T')


class Lazy(Generic[T]):
    pass


class LocalRepo:
    def __init__(self,
                 db_name: str,
                 db_username: str,
                 db_password: str,
                 db_hostname: str,
                 db_host_inbound_port: str = '3306',
                 ):
        # mysql://vegi:vegi2022!@localhost:3306/vegi
        self.mysql_conn_str = f'mysql://{db_username}:{db_password}@{db_hostname}:{db_host_inbound_port}/{db_name}'
        self.db_hostname_and_port = f'{db_hostname}:{db_host_inbound_port}'
        self.db_hostname = db_hostname
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.connection_state = 'new_instance'
        self.mysql_connection_state = 'disconnected'
        self.sftp_connection_state = 'disconnected'

        self.conn: paramiko.SFTPClient
        self.conn_db: pymysql.Connection
        
    def __enter__(self):
        self.connection_state = 'ready'
        return self

    # ...

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection_state = 'closed'
        self.close()
        
    def close(self):
        try:
            if self.sftp_connection_state != 'disconnected':
                self.conn.close()
                self.sftp_connection_state = 'disconnected'
        except Exception:
            pass
        
        try:
            if self.mysql_connection_state != 'disconnected':
                self.conn_db.close()
                self.mysql_connection_state = 'disconnected'
        except Exception:
            pass
        
    def add_MYSQL_connection(self):
        if self.mysql_connection_state == 'connected':
            return self

        conn_db: sqlite3.Connection | pymysql.Connection
        if self.mysql_conn_str.startswith('mysql'):
            self.conn_db = pymysql.connect(
                host=self.db_hostname,
                port=3306,
                user=self.db_username,
                passwd=self.db_password,
                db=self.db_name
            )
        # elif self.mysql_conn_str.lower().startswith('sql'):

        #     with self.conn.open(self.mysql_conn_str) as f:
        #         conn_db = sqlite3.connect(f.read())
        else:
            # default to mysql
            # self.add_SFTP_connection()
            with self.conn.open(self.mysql_conn_str) as f:
                self.conn_db = pymysql.connect(
                    host=self.db_hostname,
                    user=self.db_username,
                    passwd=self.db_password,
                    db=self.db_name
                )

        self.mysql_connection_state = 'connected'
        return self

    def read_all_records_from_users(self):
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        cursor.execute('SELECT u.* FROM `vegi`.user as u')
        return cursor.fetchall()
    
    def get_product_ids_that_need_rating(self, n: int = 10) -> list[int]:
        n = max(1, n)
        self.add_MYSQL_connection()
        cursor = self.conn_db.cursor()
        cursor.execute(
            'SELECT p.id FROM product as p left join escrating r on r.product = p.id where r.id IS NULL')
        ratingless_products = cursor.fetchall()
        if ratingless_products and len(ratingless_products) >= n:
            return ratingless_products[:n]
        cursor = self.conn_db.cursor()
        cursor.execute(
            'SELECT UNIQUE(p.id) FROM product as p left join escrating r on r.product = p.id order by r.calculatedOn ASC')
        old_ratings = cursor.fetchall()
        return [
            *ratingless_products,
            *old_ratings[:n]
        ]
        
    
    def get_product_to_rate(self, id: int):
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except Exception:
            return {}
        cursor.execute(
            f'SELECT p.* FROM `vegi`.product as p where p.id = {id}')
        return cursor.fetchall()

    def get_product_ratings(self, id: int):
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except Exception:
            return {}
        cursor.execute(
            select_products_sql(id))
        return cursor.fetchall()
    
    def _connect_ssh(self):
        return self


class SSHRepo:
    '''**DEPRECATED CLASS** - USE flask-sqlalchemy repo layers defined in vegi_repo.py and vegi_esc_repo.py'''
    def __init__(self,
                 server_hostname: str,
                 ssh_user: str,
                 db_name: str,
                 db_username: str,
                 db_password: str,
                 db_hostname: str = '127.0.0.1',
                 db_protocol: str = 'postgres',
                 ssh_pkey: str|None = None,
                 ssh_password: str | None = None,
                 db_host_inbound_port: str = '5432',
                 start_ssh_client: bool = True
                 ):
        '''
        Connect to SQL DB over ssh
        - hostname: str (i.e. "192.0.2.0")
        - username: str (i.e. "ubuntu")
        '''
        self.server_hostname = server_hostname
        self.ssh_user = ssh_user
        self.ssh_pkey_file = ssh_pkey
        self.ssh_password = ssh_pkey
        
        # mysql://vegi:vegi2022!@localhost:3306/vegi
        
        self.db_protocol = db_protocol
        self.mysql_conn_str = f'{db_protocol}://{db_username}:{db_password}@{db_hostname}:{db_host_inbound_port}/{db_name}'
        self.db_hostname_and_port = f'{db_hostname}:{db_host_inbound_port}'
        self.db_hostname = db_hostname
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.connection_state = 'new_instance'
        self.mysql_connection_state = 'disconnected'
        self.sftp_connection_state = 'disconnected'
        
        self.ssh: paramiko.SSHClient
        self.conn: paramiko.SFTPClient
        self.conn_db: pymysql.Connection
        if start_ssh_client:
            self.ssh = paramiko.SSHClient()
            policy = paramiko.AutoAddPolicy()
            self.ssh.set_missing_host_key_policy(policy)
                    
    def __enter__(self):
        self.connection_state = 'ready'
        return self

    # ...

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection_state = 'closed'
        self.close()
        
    def ssh_connect(self, fn: Callable):
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            self._connect_ssh()
            result = fn(*args, **kwargs)
            return result

        return wrapper
            
    def _connect_ssh(self):
        # ~ https://www.linode.com/docs/guides/use-paramiko-python-to-ssh-into-a-server/
        print('Connecting to DB Server over SSH...')
        self.ssh = paramiko.SSHClient()
        policy = paramiko.AutoAddPolicy()
        
        self.ssh.set_missing_host_key_policy(policy)
        if self.ssh_pkey_file is None:
            raise ValueError(f'Private Key filepath was not passed to [{type(self).__name__}] instance')
        # self.ssh.connect(
        #     self.server_hostname,
        #     username=self.ssh_user,
        #     key_filename=self.ssh_pkey
        #     )
        
        with open(self.ssh_pkey_file, 'r') as f:
            s = f.read()
            keyfile = io.StringIO(s)
            pkey = paramiko.RSAKey.from_private_key(keyfile)
            self.ssh.connect(self.server_hostname, username=self.ssh_user, pkey=pkey)
        # pkey = paramiko.RSAKey.from_private_key_file(self.ssh_pkey)
        # self.ssh.connect(self.server_hostname, username=self.ssh_user, pkey=pkey)
        # conn_db = pymysql.connect(
        #     host='localhost',
        #     port=3306, 
        #     user=self.db_username,
        #     passwd=self.db_password,
        #     db=self.db_name
        #     )
        self.connection_state = 'open'
        print('Connected to DB Server over SSH using pk :)')
        return self
    
    def connect_pw(self):
        # ~ https://www.linode.com/docs/guides/use-paramiko-python-to-ssh-into-a-server/
        print('Connecting to DB Server over SSH...')
        self.ssh = paramiko.SSHClient()
        policy = paramiko.AutoAddPolicy()
        self.ssh.set_missing_host_key_policy(policy)
        if self.ssh_password is None:
            raise ValueError(
                f'Password to connect to remove server over ssh was not passed to [{type(self).__name__}] instance')
        self.ssh.connect(self.server_hostname, username=self.ssh_user, password=self.ssh_password)
        
        info('Connected to DB Server over SSH using password :)')
        self.connection_state = 'open'
        return self
    
    def add_SFTP_connection(self):
        if self.sftp_connection_state == 'connected':
            return self
        
        self.conn = self.ssh.open_sftp()
        self.sftp_connection_state = 'connected'
        info('openned SFTP connection :)')
        return self

    def add_MYSQL_connection(self):
        if self.mysql_connection_state == 'connected':
            return self
        conn_db: sqlite3.Connection | pymysql.Connection
        if self.mysql_conn_str.startswith('mysql'):
            self.conn_db = pymysql.connect(
                host=self.db_hostname,
                port=3306, 
                user=self.db_username,
                passwd=self.db_password,
                db=self.db_name
                )
        # elif self.mysql_conn_str.lower().startswith('sql'):
            
        #     with self.conn.open(self.mysql_conn_str) as f:
        #         conn_db = sqlite3.connect(f.read())
        else:
            # default to mysql
            # self.add_SFTP_connection()
            with self.conn.open(self.mysql_conn_str):
                self.conn_db = pymysql.connect(
                    host=self.db_hostname,
                    user=self.db_username,
                    passwd=self.db_password,
                    db=self.db_name
                    )
            
        self.mysql_connection_state = 'connected'
        return self
    
    def read_all_records_from_users(self):
        self.add_SFTP_connection()
        self.add_MYSQL_connection()
            
        cursor = self.conn_db.cursor()
        cursor.execute('SELECT u.* FROM `vegi`.user as u')
        return cursor.fetchall()
    
    def get_product_to_rate(self, id: int):
        self.add_SFTP_connection()
        self.add_MYSQL_connection()
            
        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except Exception:
            return {}
        cursor.execute(f'SELECT p.* FROM `vegi`.product as p where p.id = {id}')
        return cursor.fetchall()
    
    def get_product_ratings(self, id: int):
        self.add_SFTP_connection()
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except Exception:
            return {}
        cursor.execute(
            select_products_sql(id))
        return cursor.fetchall()

    def close(self):
        try:
            if self.sftp_connection_state != 'disconnected':
                self.conn.close()
                self.sftp_connection_state = 'disconnected'
        except Exception:
            pass
        try:
            if self.ssh is not None:
                self.ssh.close()
        except Exception:
            pass
        try:
            if self.mysql_connection_state != 'disconnected':
                self.conn_db.close()
                self.mysql_connection_state = 'disconnected'
        except Exception:
            pass
            
    def examine_last_and_close(self, server: str):
        expected = ["user1", "reboot", "root", "sys-admin"]
        _stdin, stdout, _stderr = self.ssh.exec_command("sudo last")
        lines = stdout.read().decode()
        self.ssh.close()
        for line in lines.split("\n"):
            # Ignore the last line of the last report.
            if line.startswith("wtmp begins"):
                break
            parts = line.split()
            if parts:
                account = parts[0]
                if account not in expected:
                    warn(f"Entry '{line}' is a surprise on {server}.")


if __name__ == '__main__':
    VEGI_SERVER_P_KEY_FILE: str = ''
    # VEGI_SERVER_PUBLIC_HOSTNAME = ''
    # VEGI_SERVER_PUBLIC_IP_ADDRESS = ''
    
    # VEGI_SERVER_USERNAME = ''
    # SSH_ARGS = ''
    # SQL_USER = ''
    # SQL_PASSWORD = ''
    # SQL_DB_NAME = ''
    load_dotenv()
    config = os.environ
    if config:
        # VEGI_SERVER_IP_ADDRESS = config['DATABASE_HOST']
        # VEGI_SERVER_PUBLIC_HOSTNAME = config['DATABASE_HOST']
        # VEGI_SERVER_USERNAME = config['DATABASE_VEGI_USERNAME']
        # SQL_USER = config['DATABASE_VEGI_USERNAME']
        # SQL_PASSWORD = config['DATABASE_VEGI_PASSWORD']
        # SQL_DB_NAME = config['DATABASE_VEGI_DBNAME']
        # VEGI_SERVER_P_KEY_FILE = config.get('ansible_ssh_private_key_file','').replace('~', '/Users/joey')
        VEGI_SERVER_P_KEY_FILE = os.path.expanduser(
            config.get('ansible_ssh_private_key_file', ''))
        SSH_ARGS = config.get('ansible_ssh_extra_args', '')
    # import yaml
    # with open('hosts.yml') as f:
    #     hostKVPs = yaml.load(f, Loader=yaml.FullLoader)
        # useQA = False
        # if useQA:
        #     config = hostKVPs['all']['hosts']['vegi-backend-qa']
        #     VEGI_SERVER_IP_ADDRESS = config['ansible_ssh_host']
        
        #     VEGI_SERVER_PUBLIC_HOSTNAME = 'ec2-'+config['ansible_ssh_host'].replace(
        #         '.', '-')+'.compute-1.amazonaws.com'  # ec2-54-221-0-234.compute-1.amazonaws.com
        #     VEGI_SERVER_USERNAME = config['ansible_user']
        #     # VEGI_SERVER_P_KEY_FILE = config['ansible_ssh_private_key_file'].replace('~', '/Users/joey')
        #     VEGI_SERVER_P_KEY_FILE = os.path.expanduser(
        #         config['ansible_ssh_private_key_file'])
        #     SSH_ARGS = config['ansible_ssh_extra_args']
        # else:
        #     config = hostKVPs['all']['hosts']['vegi-localhost']
        
        # SQL_USER = config['mysql_production_user']
        # SQL_PASSWORD = config['mysql_production_password']
        # SQL_DB_NAME = config['mysql_production_database']
    
    if VEGI_SERVER_P_KEY_FILE:
        with SSHRepo(
            server_hostname=config['DATABASE_HOST'],
            ssh_user=config['SERVER_USERNAME'],
            ssh_pkey=VEGI_SERVER_P_KEY_FILE,
            db_hostname=config['DATABASE_HOST'],
            db_host_inbound_port=config['DATABASE_PORT'],
            db_name=config['DATABASE_VEGI_DBNAME'],
            db_username=config['DATABASE_VEGI_USERNAME'],
            db_password=config['DATABASE_VEGI_PASSWORD'],
        ) as repoConn:
            users = repoConn.read_all_records_from_users()

            # repoConn.close()

            info(users)
    else:
        with LocalRepo(
            db_hostname=config['DATABASE_HOST'], # localhost
            db_host_inbound_port=config['DATABASE_PORT'],
            db_name=config['DATABASE_VEGI_DBNAME'],
            db_username=config['DATABASE_VEGI_USERNAME'],
            db_password=config['DATABASE_VEGI_PASSWORD'],
        ) as repoConn:

            users = repoConn.read_all_records_from_users()

            # repoConn.close()

            info(users)

    info('DONE')


# def appcontext(f: Callable):
#     def deco(self: VegiRepo, *args: Any, **kwargs: Any):
#         with self.app.app_context():
#             return f(self, *args, **kwargs)
#     return deco


# class VegiRepo():
#     def __init__(self,
#                  app: Flask,
#                  env: str = 'QA',
#                  start_ssh_client: bool = True,
#                  ):
#         '''
#         Connect to SQL DB over ssh if db hosted in heroku
#         - env: str (i.e. "QA" for heroku QA instance and "DEV" for localhost)
#         '''
#         VegiRepo.app = app
#         self.env = env
#         self.use_ssh = env == 'QA'
#         self.start_ssh_client = start_ssh_client
                    
#     def __enter__(self):
#         self.connection_state = 'ready'
#         load_dotenv()
#         config = os.environ
#         if self.env == 'QA':
#             with SSHRepo(
#                 server_hostname=config['DATABASE_HOST'],
#                 ssh_user=config['SERVER_USERNAME'],
#                 ssh_pkey=VEGI_SERVER_P_KEY_FILE,
#                 db_hostname=config['DATABASE_HOST'],
#                 db_host_inbound_port=config['DATABASE_PORT'],
#                 db_name=config['DATABASE_VEGI_DBNAME'],
#                 db_username=config['DATABASE_VEGI_USERNAME'],
#                 db_password=config['DATABASE_VEGI_PASSWORD'],
#                 start_ssh_client=self.start_ssh_client,
#             ) as repoConn:
#                 self.repoConn = repoConn
#                 return self.repoConn
#         else:
#             with LocalRepo(
#                 db_hostname=config['DATABASE_HOST'],  # localhost
#                 db_host_inbound_port=config['DATABASE_PORT'],
#                 db_name=config['DATABASE_VEGI_DBNAME'],
#                 db_username=config['DATABASE_VEGI_USERNAME'],
#                 db_password=config['DATABASE_VEGI_PASSWORD'],
#             ) as repoConn:
#                 self.repoConn = repoConn
#                 return self.repoConn

#     # ...

#     def __exit__(self, exc_type, exc_value, traceback):
#         self.connection_state = 'closed'
#         self.repoConn.close()
        
#     def connect_ssh(self):
#         if self.use_ssh:
#             self.repoConn._connect_ssh()
            
#     @appcontext
#     def get_sources(self, source_type: str | None = None):
#         try:
#             sources: list[ESCSource] = (
#                 ESCSource.query.all()
#                 if source_type
#                 else ESCSource.query.filter(ESCSource.source_type == source_type).all()
#             )
#             assert isinstance(sources, list)
#             return [e.serialize() for e in sources]
#         except Exception as e:
#             logger.error(str(e))
#             return []