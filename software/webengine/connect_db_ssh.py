
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/09/2022
#############################################################################

import mysql.connector
import pandas as pd
import os
import paramiko

def connect_to_DB(MainWindow):
    ssh_db_info = pd.read_csv(os.path.join(MainWindow.wd, 'webengine', 'ssh_database.info'), index_col=0)
    DB_HOST = ssh_db_info.loc['db_IP','value']
    DB_USER = ssh_db_info.loc['db_user','value']
    DB_PASS = ssh_db_info.loc['db_pwd','value']
    DB_DBNAME = ssh_db_info.loc['db_dbname','value']

    mydb = mysql.connector.connect(
      host=DB_HOST,
      user=DB_USER,
      password=DB_PASS,
      database=DB_DBNAME
    )
    return mydb

def connect_to_SSH(MainWindow):
    ssh_db_info = pd.read_csv(os.path.join(MainWindow.wd, 'webengine', 'ssh_database.info'), index_col=0)
    SSH_HOST = ssh_db_info.loc['ssh_IP','value']
    SSH_USER = ssh_db_info.loc['ssh_user','value']
    SSH_PASS = ssh_db_info.loc['ssh_pwd','value']
    REMOTE_DIR = ssh_db_info.loc['remote_path','value']

    cli = paramiko.client.SSHClient()
    cli.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
    cli.connect(hostname=SSH_HOST, username=SSH_USER, password=SSH_PASS)
    return cli, REMOTE_DIR
