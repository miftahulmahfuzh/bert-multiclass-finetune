#!/bin/bash

# Variables
SSH_USER="devmiftahul"
SSH_HOST="10.183.0.2"
SSH_PORT=2288
LOCAL_PORT=11435
REMOTE_PORT=11435
PASSWORD="RvVK6!AmsW"

# Establish SSH Tunnel using sshpass
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $SSH_PORT -L $LOCAL_PORT:localhost:$REMOTE_PORT $SSH_USER@$SSH_HOST
