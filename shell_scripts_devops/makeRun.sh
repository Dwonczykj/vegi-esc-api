#! /bin/bash

make miniconda
$imageName=vegi_esc_server-miniconda
docker run --platform linux/amd64 -it -p 2001:5002 $imageName
open localhost:2001/success/fenton