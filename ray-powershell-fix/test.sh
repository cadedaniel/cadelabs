#!/usr/bin/env bash

pwsh=$(which pwsh)

export COMPLEX_ENV_VAR_1=$(cat <<- "EOF"
[Jobs] Deflake jobs sdk test (#29707)
--
Wait for all nodes to come up before running the unit test to deflake the unit test. It submits a bunch of jobs and checks that at least one was run on a worker node. However if the jobs are submitted too soon, its' possible that only the head node has come up, so 'theyll all be scheduled on the head node. The fix is to only submit the jobs after all the nodes are up.
Also adds some logs to improve debuggability.

Related issue number
Closes #29006
EOF)

env -i COMPLEX_ENV_VAR_1="$COMPLEX_ENV_VAR_1" $pwsh ps.ps | sed 's/export PATH/export FAKE_PATH/g' > refreshenv.sh
echo 'env' >> refreshenv.sh

env -i bash refreshenv.sh
