#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
export PYTHONPATH=$SCRIPTPATH/..:$PYTHONPATH
cd $SCRIPTPATH/../../
echo `pwd`
adev runserver nohomers --root nohomers --port 9000
