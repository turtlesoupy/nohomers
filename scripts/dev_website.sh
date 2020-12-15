#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$SCRIPTPATH/../:$PYTHONPATH
cd $SCRIPTPATH/../
echo `pwd`
echo $SCRIPTPATH
adev runserver . --port 9000
