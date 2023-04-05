# from paramiko.util import ClosingContextManager
# from paramiko.transport import Transport
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
from logger import info, LOG_LEVEL, warn
import socket
import time
import boto3
import io
import sqlite3
import pymysql

import os
from typing import Callable, TypeVar, Generic, Any

from select_vendor_products_with_esc import select_products_sql

from functools import wraps

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
        except:
            pass
        
        try:
            if self.mysql_connection_state != 'disconnected':
                self.conn_db.close()
                self.mysql_connection_state = 'disconnected'
        except:
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
    
    def get_product_to_rate(self, id: int):
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except:
            return {}
        cursor.execute(
            f'SELECT p.* FROM `vegi`.product as p where p.id = {id}')
        return cursor.fetchall()

    def get_product_ratings(self, id: int):
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except:
            return {}
        cursor.execute(
            select_products_sql(id))
        return cursor.fetchall()


class SSHRepo:
    def __init__(self, 
                 server_hostname:str,
                 ssh_user:str,
                 db_name: str,
                 db_username:str,
                 db_password:str,
                 db_hostname:str,
                 ssh_pkey:str|None=None,
                 ssh_password:str|None=None,
                 db_host_inbound_port:str='3306',
                 start_ssh_client:bool=True
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
        self.mysql_conn_str = f'mysql://{db_username}:{db_password}@{db_hostname}:{db_host_inbound_port}/{db_name}'
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
        
        conn_db:sqlite3.Connection|pymysql.Connection
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
        self.add_SFTP_connection()
        self.add_MYSQL_connection()
            
        cursor = self.conn_db.cursor()
        cursor.execute('SELECT u.* FROM `vegi`.user as u')
        return cursor.fetchall()
    
    def get_product_to_rate(self, id:int):
        self.add_SFTP_connection()
        self.add_MYSQL_connection()
            
        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except:
            return {}
        cursor.execute(f'SELECT p.* FROM `vegi`.product as p where p.id = {id}')
        return cursor.fetchall()
    
    def get_product_ratings(self, id: int):
        self.add_SFTP_connection()
        self.add_MYSQL_connection()

        cursor = self.conn_db.cursor()
        try:
            id = int(id)
        except:
            return {}
        cursor.execute(
            select_products_sql(id))
        return cursor.fetchall()

    def close(self):
        try:
            if self.sftp_connection_state != 'disconnected':
                self.conn.close()
                self.sftp_connection_state = 'disconnected'
        except:
            pass
        try:
            if self.ssh is not None:
                self.ssh.close()
        except:
            pass
        try:
            if self.mysql_connection_state != 'disconnected':
                self.conn_db.close()
                self.mysql_connection_state = 'disconnected'
        except:
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
                if not account in expected:
                    warn(f"Entry '{line}' is a surprise on {server}.")

if __name__ == '__main__':
    import yaml
    VEGI_SERVER_P_KEY_FILE: str = ''
    VEGI_SERVER_PUBLIC_HOSTNAME = ''
    VEGI_SERVER_PUBLIC_IP_ADDRESS = ''
    VEGI_SERVER_PRIVATE_IP_ADDRESS = ''
    VEGI_SERVER_USERNAME = ''
    SSH_ARGS = ''
    SQL_USER = ''
    SQL_PASSWORD = ''
    SQL_DB_NAME = ''
    with open('hosts.yml') as f:
        hostKVPs = yaml.load(f, Loader=yaml.FullLoader)
        useQA = False
        if useQA:
            config = hostKVPs['all']['hosts']['vegi-backend-qa']
            VEGI_SERVER_IP_ADDRESS = config['ansible_ssh_host']
            VEGI_SERVER_PRIVATE_IP_ADDRESS = config['ansible_ssh_private_ip']
            VEGI_SERVER_PUBLIC_HOSTNAME = 'ec2-'+config['ansible_ssh_host'].replace(
                '.', '-')+'.compute-1.amazonaws.com'  # ec2-54-221-0-234.compute-1.amazonaws.com
            VEGI_SERVER_USERNAME = config['ansible_user']
            # VEGI_SERVER_P_KEY_FILE = config['ansible_ssh_private_key_file'].replace('~', '/Users/joey')
            VEGI_SERVER_P_KEY_FILE = os.path.expanduser(
                config['ansible_ssh_private_key_file'])
            SSH_ARGS = config['ansible_ssh_extra_args']
        else:
            config = hostKVPs['all']['hosts']['vegi-localhost']
        
        SQL_USER = config['mysql_production_user']
        SQL_PASSWORD = config['mysql_production_password']
        SQL_DB_NAME = config['mysql_production_database']
    
    if VEGI_SERVER_P_KEY_FILE:
        with SSHRepo(
            # server_hostname=VEGI_SERVER_PRIVATE_IP_ADDRESS,
            server_hostname=VEGI_SERVER_PUBLIC_HOSTNAME,
            ssh_user=VEGI_SERVER_USERNAME,
            ssh_pkey=VEGI_SERVER_P_KEY_FILE,
            db_hostname='localhost',
            db_host_inbound_port='3306',
            db_name=SQL_DB_NAME,
            db_username=SQL_USER,
            db_password=SQL_PASSWORD,
        ) as repoConn:

            

            users = repoConn.read_all_records_from_users()

            # repoConn.close()

            info(users)
    else:
        with LocalRepo(
            db_hostname='localhost',
            db_host_inbound_port='3306',
            db_name=SQL_DB_NAME,
            db_username=SQL_USER,
            db_password=SQL_PASSWORD,
        ) as repoConn:

            users = repoConn.read_all_records_from_users()

            # repoConn.close()

            info(users)
        

    info('DONE')